"""Segmentation models of A2, written from the papers.

  SegFormer        (Xie et al., NeurIPS 2021): hierarchical Mix Transformer encoder (MiT) + all-MLP decoder
  DeepLabV3+       (Chen et al., ECCV 2018):   dilated ResNet-50 + ASPP + low-level skip decoder
  ResNet-50 + MLP  : the SegFormer decoder on a CNN encoder (the paper's Table 1d ablation)

Every model maps a normalised image [B,3,H,W] to logits [B,num_classes,H/4,W/4]; the training and
evaluation code upsamples the logits bilinearly to the label resolution.

The MiT/decoder modules use the parameter names of the Hugging Face `transformers` port so that the
official ImageNet-1k encoders (`nvidia/mit-b*`) and fine-tuned checkpoints load without key remapping
(see scripts/verify_segformer.py for the numerical check against the reference implementation).
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- SegFormer: MiT encoder
MIT_CONFIGS = {
    #        embed dims             depths        decoder dim
    "b0": ((32, 64, 160, 256), (2, 2, 2, 2), 256),
    "b1": ((64, 128, 320, 512), (2, 2, 2, 2), 256),
    "b2": ((64, 128, 320, 512), (3, 4, 6, 3), 768),
}
MIT_HEADS = (1, 2, 5, 8)
MIT_SR_RATIOS = (8, 4, 2, 1)     # spatial reduction of keys/values per stage
MIT_PATCH = ((7, 4), (3, 2), (3, 2), (3, 2))  # (kernel, stride) of the overlapped patch merging
MIT_MLP_RATIO = 4


class DropPath(nn.Module):
    """Stochastic depth: drop the whole residual branch of a sample with probability p."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(1.0 - self.p)
        return x * keep / (1.0 - self.p)


class OverlapPatchEmbed(nn.Module):
    """Overlapped patch merging: a strided conv with kernel > stride, then LayerNorm on tokens."""

    def __init__(self, in_ch, dim, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel, stride, kernel // 2)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        h, w = x.shape[-2:]
        return self.layer_norm(x.flatten(2).transpose(1, 2)), h, w


class EfficientSelfAttention(nn.Module):
    """Self-attention whose keys/values come from a sequence reduced R^2 times (a strided RxR conv),
    so the cost drops from O(N^2) to O(N^2 / R^2)."""

    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        q = self.query(x).view(b, n, self.heads, c // self.heads).transpose(1, 2)
        if self.sr_ratio > 1:
            x = self.sr(x.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
            x = self.layer_norm(x)
        k = self.key(x).view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        v = self.value(x).view(b, -1, self.heads, c // self.heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, n, c)


class AttentionOutput(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)

    def forward(self, x):
        return self.dense(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.self = EfficientSelfAttention(dim, heads, sr_ratio)
        self.output = AttentionOutput(dim)

    def forward(self, x, h, w):
        return self.output(self.self(x, h, w))


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, h, w):
        b, n, c = x.shape
        x = self.dwconv(x.transpose(1, 2).reshape(b, c, h, w))
        return x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    """Feed-forward network with a 3x3 depth-wise conv between the two linear layers. The zero
    padding of the conv leaks position information, which replaces the positional encoding."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.dense1 = nn.Linear(dim, hidden)
        self.dwconv = DWConv(hidden)
        self.dense2 = nn.Linear(hidden, dim)

    def forward(self, x, h, w):
        return self.dense2(F.gelu(self.dwconv(self.dense1(x), h, w)))


class MiTBlock(nn.Module):
    def __init__(self, dim, heads, sr_ratio, drop_path):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = Attention(dim, heads, sr_ratio)
        self.drop_path = DropPath(drop_path)
        self.layer_norm_2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MixFFN(dim, dim * MIT_MLP_RATIO)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attention(self.layer_norm_1(x), h, w))
        return x + self.drop_path(self.mlp(self.layer_norm_2(x), h, w))


class MiT(nn.Module):
    """Mix Transformer encoder: 4 stages with strides 4/8/16/32 and no positional embedding."""

    def __init__(self, variant="b0", drop_path_rate=0.1):
        super().__init__()
        dims, depths, _ = MIT_CONFIGS[variant]
        self.channels = dims
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.patch_embeddings = nn.ModuleList()
        self.block = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.patch_embeddings.append(OverlapPatchEmbed(3 if i == 0 else dims[i - 1], dims[i], *MIT_PATCH[i]))
            self.block.append(nn.ModuleList(
                MiTBlock(dims[i], MIT_HEADS[i], MIT_SR_RATIOS[i], rates[cur + j]) for j in range(depths[i])))
            self.layer_norm.append(nn.LayerNorm(dims[i], eps=1e-6))
            cur += depths[i]
        self.apply(_init_mit)

    def forward(self, x):
        feats = []
        for embed, blocks, norm in zip(self.patch_embeddings, self.block, self.layer_norm):
            x, h, w = embed(x)
            for blk in blocks:
                x = blk(x, h, w)
            # tokens [B,N,C] -> map [B,C,H,W]; contiguous() matters: without it the map keeps a
            # channels-last layout and the conv/BatchNorm layers downstream take a slow path
            x = norm(x).transpose(1, 2).reshape(x.shape[0], -1, h, w).contiguous()
            feats.append(x)
        return feats


def _init_mit(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ----------------------------------------------------------------------------- SegFormer: MLP decoder
class MLPProj(nn.Module):
    def __init__(self, in_ch, dim):
        super().__init__()
        self.proj = nn.Linear(in_ch, dim)

    def forward(self, x):  # [B,C,H,W] -> [B,dim,H,W]
        return self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MLPDecoder(nn.Module):
    """All-MLP decoder: unify the channels of the 4 feature maps (linear_c), upsample them to 1/4
    resolution, concatenate, fuse with one 1x1 conv (linear_fuse), then BN-ReLU and classify.
    Encoder-agnostic.

    Two forward paths give the same output:
      * the literal one of the paper (`literal=True`): it materialises a [B, 4C, H/4, W/4] tensor,
        which costs ~3 GB at C=768, 512x512, batch 8;
      * the default one: projection, bilinear upsampling and the 1x1 fusion are all linear with no
        non-linearity in between, so   fuse(cat_i up(P_i f_i)) = sum_i up((W_i P_i) f_i).
        The two linear maps of each stage are merged into one 1x1 conv applied at the native
        resolution of that stage and the upsampled results are summed. Parameters are unchanged
        (official checkpoints still load); memory and compute drop sharply.
    """

    def __init__(self, in_channels, dim, num_classes, dropout=0.1, literal=False):
        super().__init__()
        self.dim, self.literal = dim, literal
        self.linear_c = nn.ModuleList(MLPProj(c, dim) for c in in_channels)
        self.linear_fuse = nn.Conv2d(dim * len(in_channels), dim, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(dim, num_classes, 1)

    def fuse(self, feats):
        size = feats[0].shape[-2:]
        if self.literal:
            ups = [F.interpolate(proj(f).contiguous(), size=size, mode="bilinear", align_corners=False)
                   for proj, f in zip(self.linear_c, feats)]
            return self.linear_fuse(torch.cat(ups[::-1], dim=1))
        n, out = len(feats), 0
        w_fuse = self.linear_fuse.weight.float().flatten(1)          # [C, n*C]; stage i uses block n-1-i
        for i, (proj, f) in enumerate(zip(self.linear_c, feats)):
            with torch.autocast("cuda", enabled=False):               # merge the weights in fp32
                w_i = w_fuse[:, (n - 1 - i) * self.dim:(n - i) * self.dim]
                w = w_i @ proj.proj.weight.float()                    # [C, C_i]
                b = w_i @ proj.proj.bias.float()
            y = F.conv2d(f, w[:, :, None, None], b)                   # runs in the autocast dtype
            out = out + (y if i == 0 else F.interpolate(y, size=size, mode="bilinear", align_corners=False))
        return out

    def forward(self, feats):
        x = F.relu(self.batch_norm(self.fuse(feats)))
        return self.classifier(self.dropout(x))


# ----------------------------------------------------------------------------- DeepLabV3+ head
def conv_bn_relu(in_ch, out_ch, kernel=1, dilation=1, separable=False):
    pad = dilation * (kernel // 2)
    if separable:  # depth-wise 3x3 (atrous) followed by point-wise 1x1, as in the DeepLabV3+ paper
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel, padding=pad, dilation=dilation, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel, padding=pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: parallel atrous convs with several rates + image pooling."""

    def __init__(self, in_ch, dim=256, rates=(6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_relu(in_ch, dim, 1)])
        self.branches.extend(conv_bn_relu(in_ch, dim, 3, r, separable=True) for r in rates)
        # image-level branch: no BatchNorm, because its 1x1 map has a single value per channel
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, dim, 1), nn.ReLU(inplace=True))
        self.project = conv_bn_relu(dim * (len(rates) + 2), dim, 1)

    def forward(self, x):
        outs = [b(x) for b in self.branches]
        outs.append(self.image_pool(x).expand(-1, -1, *x.shape[-2:]))
        return self.project(torch.cat(outs, dim=1))


class DeepLabV3PlusHead(nn.Module):
    def __init__(self, in_channels, num_classes, dim=256, low_dim=48, dropout=0.1):
        super().__init__()
        self.aspp = ASPP(in_channels[3], dim)
        self.low_proj = conv_bn_relu(in_channels[0], low_dim, 1)
        self.fuse = nn.Sequential(conv_bn_relu(dim + low_dim, dim, 3, separable=True),
                                  conv_bn_relu(dim, dim, 3, separable=True))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Conv2d(dim, num_classes, 1)

    def forward(self, feats):
        low = self.low_proj(feats[0])                                  # stride 4
        x = F.interpolate(self.aspp(feats[3]), size=low.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.dropout(self.fuse(torch.cat([x, low], dim=1))))


# ----------------------------------------------------------------------------- encoders / full models
class ResNetEncoder(nn.Module):
    """timm ResNet-50 returning the 4 stage outputs. output_stride=16 dilates the last stage."""

    def __init__(self, timm_name, pretrained, output_stride):
        super().__init__()
        self.body = timm.create_model(timm_name, pretrained=pretrained, features_only=True,
                                      out_indices=(1, 2, 3, 4), output_stride=output_stride)
        self.channels = tuple(self.body.feature_info.channels())

    def forward(self, x):
        return self.body(x)


class SegModel(nn.Module):
    def __init__(self, encoder, decode_head):
        super().__init__()
        self.encoder = encoder
        self.decode_head = decode_head

    def forward(self, x):
        return self.decode_head(self.encoder(x))


RESNET50 = "resnet50.tv2_in1k"  # same ImageNet-1k checkpoint as in A1
MODELS = {
    "deeplabv3plus_r50": {"family": "CNN", "label": "DeepLabV3+ R50", "encoder": "resnet50", "decoder": "aspp",
                          "pretrained_id": RESNET50},
    "resnet50_mlp": {"family": "CNN", "label": "ResNet-50 + MLP decoder", "encoder": "resnet50", "decoder": "mlp",
                     "pretrained_id": RESNET50},
    "segformer_b0": {"family": "Transformer", "label": "SegFormer-B0", "encoder": "b0", "decoder": "mlp",
                     "pretrained_id": "nvidia/mit-b0"},
    "segformer_b1": {"family": "Transformer", "label": "SegFormer-B1", "encoder": "b1", "decoder": "mlp",
                     "pretrained_id": "nvidia/mit-b1"},
    "segformer_b2": {"family": "Transformer", "label": "SegFormer-B2", "encoder": "b2", "decoder": "mlp",
                     "pretrained_id": "nvidia/mit-b2"},
}


def load_hf_state_dict(repo_id):
    """Download a Hugging Face checkpoint and return its state dict (safetensors if available)."""
    from huggingface_hub import hf_hub_download
    try:
        from safetensors.torch import load_file
        return load_file(hf_hub_download(repo_id, "model.safetensors"))
    except Exception:
        return torch.load(hf_hub_download(repo_id, "pytorch_model.bin"), map_location="cpu", weights_only=True)


def load_mit_pretrained(encoder, variant):
    """Load the official ImageNet-1k MiT encoder (the 1000-class head of the checkpoint is dropped)."""
    sd = load_hf_state_dict(f"nvidia/mit-{variant}")
    prefix = "segformer.encoder."
    sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = encoder.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def build_model(arch, num_classes=21, pretrained=True, decoder_dim=None, drop_path_rate=0.1):
    cfg = MODELS[arch]
    if cfg["encoder"] == "resnet50":
        if cfg["decoder"] == "aspp":
            enc = ResNetEncoder(RESNET50, pretrained, output_stride=16)
            head = DeepLabV3PlusHead(enc.channels, num_classes)
        else:
            enc = ResNetEncoder(RESNET50, pretrained, output_stride=32)
            head = MLPDecoder(enc.channels, decoder_dim or 768, num_classes)
    else:
        variant = cfg["encoder"]
        enc = MiT(variant, drop_path_rate)
        if pretrained:
            load_mit_pretrained(enc, variant)
        head = MLPDecoder(enc.channels, decoder_dim or MIT_CONFIGS[variant][2], num_classes)
    return SegModel(enc, head)


def load_trained(run, ckpt_dir="checkpoints", device="cuda"):
    """Rebuild a model trained by src.train from checkpoints/<run>/final.pt (eval mode)."""
    import os
    state = torch.load(os.path.join(ckpt_dir, run, "final.pt"), map_location="cpu", weights_only=True)
    model = build_model(state["arch"], pretrained=False, decoder_dim=state["decoder_dim"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def count_params(model):
    enc = sum(p.numel() for p in model.encoder.parameters())
    head = sum(p.numel() for p in model.decode_head.parameters())
    return enc + head, enc, head

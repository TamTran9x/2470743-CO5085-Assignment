"""Model zoo (one CNN, one ViT) and the four training regimes.

Regimes
  scratch : random init, everything trainable
  linear  : ImageNet-1k pretrained, whole backbone frozen, only the new head is trained
  partial : pretrained, first half of the network frozen (ResNet stem+layer1-2 / ViT patch-embed+blocks 0-5)
  full    : pretrained, everything fine-tuned
"""
import timm

# Both checkpoints are trained on ImageNet-1k only, with similar size (22-26M params, ~4-5 GFLOPs @224)
ARCHS = {
    "resnet50": {"timm_name": "resnet50.tv2_in1k", "family": "CNN"},
    "deit_s": {"timm_name": "deit_small_patch16_224.fb_in1k", "family": "Transformer"},
}
MODES = ("scratch", "linear", "partial", "full")


def _frozen_prefixes(arch, mode):
    if mode in ("scratch", "full"):
        return []
    if arch == "resnet50":
        if mode == "linear":
            return ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        return ["conv1", "bn1", "layer1", "layer2"]
    if arch == "deit_s":
        if mode == "linear":
            return ["cls_token", "pos_embed", "patch_embed", "blocks", "norm"]
        return ["cls_token", "pos_embed", "patch_embed"] + [f"blocks.{i}" for i in range(6)]
    raise ValueError(arch)


def _matches(name, prefixes):
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def build_model(arch, mode, num_classes=100, img_size=224):
    """Create the model and freeze parameters according to `mode`.

    Returns (model, frozen_modules); frozen modules must be kept in eval mode during
    training so BatchNorm statistics / dropout of the frozen part do not change.
    """
    assert mode in MODES, mode
    kwargs = {"pretrained": mode != "scratch", "num_classes": num_classes}
    if arch == "deit_s":
        kwargs["img_size"] = img_size  # pretrained pos-embed is resampled by timm if needed
    model = timm.create_model(ARCHS[arch]["timm_name"], **kwargs)

    prefixes = _frozen_prefixes(arch, mode)
    for name, p in model.named_parameters():
        if _matches(name, prefixes):
            p.requires_grad_(False)
    frozen_modules = [m for name, m in model.named_modules() if name and _matches(name, prefixes)]
    return model, frozen_modules


def set_train_mode(model, frozen_modules):
    model.train()
    for m in frozen_modules:
        m.eval()


def head_parameters(model):
    return list(model.get_classifier().parameters())


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def is_cnn(arch):
    return ARCHS[arch]["family"] == "CNN"

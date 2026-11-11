"""CIFAR-100 loading, stratified train/val split and a GPU-resident data pipeline.

The whole dataset (60k images of 32x32x3 uint8, ~180 MB) fits in GPU memory, so
batches are sliced, augmented and resized directly on the GPU. This avoids the
slow multi-process DataLoader on Windows and keeps the GPU saturated.
"""
import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Official CIFAR-100 (Krizhevsky, 2009) mirrored on the Hugging Face Hub; same images and
# train/test split as cifar-100-python.tar.gz, which downloads very slowly from cs.toronto.edu.
HF_REPO = "uoft-cs/cifar100"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Label names in the order of the original `meta` file
CLASSES = (
    "apple aquarium_fish baby bear beaver bed bee beetle bicycle bottle bowl boy bridge bus butterfly "
    "camel can castle caterpillar cattle chair chimpanzee clock cloud cockroach couch crab crocodile cup "
    "dinosaur dolphin elephant flatfish forest fox girl hamster house kangaroo keyboard lamp lawn_mower "
    "leopard lion lizard lobster man maple_tree motorcycle mountain mouse mushroom oak_tree orange orchid "
    "otter palm_tree pear pickup_truck pine_tree plain plate poppy porcupine possum rabbit raccoon ray road "
    "rocket rose sea seal shark shrew skunk skyscraper snail snake spider squirrel streetcar sunflower "
    "sweet_pepper table tank telephone television tiger tractor train trout tulip turtle wardrobe whale "
    "willow_tree wolf woman worm"
).split()
SUPERCLASSES = (
    "aquatic_mammals fish flowers food_containers fruit_and_vegetables household_electrical_devices "
    "household_furniture insects large_carnivores large_man-made_outdoor_things "
    "large_natural_outdoor_scenes large_omnivores_and_herbivores medium_mammals "
    "non-insect_invertebrates people reptiles small_mammals trees vehicles_1 vehicles_2"
).split()


def _download_to_npz(root, cache):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    arrays = {}
    for split in ("train", "test"):
        path = hf_hub_download(HF_REPO, f"cifar100/{split}-00000-of-00001.parquet", repo_type="dataset",
                               local_dir=os.path.join(root, "hf"))
        table = pq.read_table(path).to_pydict()
        arrays[f"x_{split}"] = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB"))
                                         for d in table["img"]])
        arrays[f"y_{split}"] = np.array(table["fine_label"], dtype=np.int64)
        arrays[f"c_{split}"] = np.array(table["coarse_label"], dtype=np.int64)
    np.savez(cache, **arrays)


def load_cifar100(root="data"):
    """Return dict with train/test arrays (NHWC uint8), fine/coarse labels and class names."""
    cache = os.path.join(root, "cifar100.npz")
    if not os.path.isfile(cache):
        os.makedirs(root, exist_ok=True)
        _download_to_npz(root, cache)
    d = dict(np.load(cache))
    assert d["x_train"].shape == (50000, 32, 32, 3) and d["x_test"].shape == (10000, 32, 32, 3)
    # fine -> coarse mapping (each fine class belongs to exactly one superclass)
    fine_to_coarse = np.zeros(100, dtype=np.int64)
    fine_to_coarse[d["y_train"]] = d["c_train"]
    d.update(classes=CLASSES, superclasses=SUPERCLASSES, fine_to_coarse=fine_to_coarse)
    return d


def stratified_split(labels, val_per_class=50, seed=0):
    """Split indices so that every class keeps exactly `val_per_class` images for validation."""
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        val_idx.append(idx[:val_per_class])
        train_idx.append(idx[val_per_class:])
    return np.sort(np.concatenate(train_idx)), np.sort(np.concatenate(val_idx))


def random_crop_flip(x, pad=4):
    """Per-sample random crop (zero padding `pad`) + horizontal flip on a uint8 NCHW batch."""
    b, _, h, w = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    iy = oy[:, None] + torch.arange(h, device=x.device)
    ix = ox[:, None] + torch.arange(w, device=x.device)
    bi = torch.arange(b, device=x.device)[:, None, None]
    out = xp[bi, :, iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)  # B,C,H,W
    flip = torch.rand(b, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class GPULoader:
    """Iterate over (images, labels) batches that already live on the GPU."""

    def __init__(self, images, labels, batch_size, img_size, train, device="cuda",
                 channels_last=False):
        self.x = torch.from_numpy(np.ascontiguousarray(images)).to(device).permute(0, 3, 1, 2).contiguous()
        self.y = torch.from_numpy(np.asarray(labels)).long().to(device)
        self.batch_size = batch_size
        self.img_size = img_size
        self.train = train
        self.channels_last = channels_last
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __len__(self):
        n = len(self.y)
        return n // self.batch_size if self.train else -(-n // self.batch_size)

    def preprocess(self, xb):
        xb = xb.float().div_(255.0)
        if self.img_size != xb.shape[-1]:
            xb = F.interpolate(xb, size=(self.img_size, self.img_size), mode="bilinear",
                               align_corners=False)
        xb = (xb - self.mean) / self.std
        if self.channels_last:
            xb = xb.contiguous(memory_format=torch.channels_last)
        return xb

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n, device=self.y.device) if self.train else torch.arange(n, device=self.y.device)
        for i in range(len(self)):
            idx = order[i * self.batch_size:(i + 1) * self.batch_size]
            xb = self.x[idx]
            if self.train:
                xb = random_crop_flip(xb)
            yield self.preprocess(xb), self.y[idx]

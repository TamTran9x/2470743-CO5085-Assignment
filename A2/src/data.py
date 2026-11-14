"""Pascal VOC 2012 (+SBD) semantic segmentation: Dataset, joint image/mask augmentation, loaders.

Training samples are decoded and augmented on the CPU by DataLoader workers (the images have
different sizes, so they cannot be stacked on the GPU like CIFAR in A1). Evaluation sets are
small (<= 1,449 images), so they are decoded once, padded to 512x512 and kept in RAM.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision.transforms import v2

ROOT = "data/voc"
CLASSES = ("background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
           "tvmonitor")
NUM_CLASSES = len(CLASSES)
IGNORE = 255                       # void label: object borders / "difficult" regions in VOC, and padding
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PAD_RGB = tuple(round(255 * m) for m in IMAGENET_MEAN)  # padding colour = 0 after normalisation
EVAL_SIZE = 512                    # every VOC image is at most 500x500


def voc_palette():
    """The standard VOC colour map (index -> RGB), 256 entries."""
    pal = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        pal[i] = (r, g, b)
    pal[IGNORE] = (255, 255, 255)
    return pal


def read_list(split, root=ROOT):
    with open(os.path.join(root, "lists", split + ".txt")) as f:
        return [tuple(line.split()) for line in f if line.strip()]


def load_pair(root, image_id, label_dir):
    img = Image.open(os.path.join(root, "JPEGImages", image_id + ".jpg")).convert("RGB")
    mask = Image.open(os.path.join(root, label_dir, image_id + ".png"))  # mode P or L: pixel = class index
    return img, mask


# ----------------------------------------------------------------------------- training pipeline
class TrainTransform:
    """Random scale -> pad -> random crop -> horizontal flip -> colour jitter (SegFormer/mmseg recipe).

    Geometric steps are applied jointly to the image (bilinear) and the mask (nearest); padding
    uses the mean colour for the image and the void label for the mask, so it never enters the loss.
    """

    def __init__(self, crop=512, scale=(0.5, 2.0), cat_max_ratio=0.75, color=True):
        self.crop, self.scale, self.cat_max_ratio = crop, scale, cat_max_ratio
        self.jitter = v2.ColorJitter(brightness=0.125, contrast=0.5, saturation=0.5, hue=0.05) if color else None

    def __call__(self, img, mask):
        s = torch.empty(1).uniform_(*self.scale).item()
        w, h = img.size
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        img, mask = np.asarray(img), np.asarray(mask)

        c = self.crop
        ph, pw = max(c - nh, 0), max(c - nw, 0)
        if ph or pw:
            img = np.stack([np.pad(img[..., k], ((0, ph), (0, pw)), constant_values=PAD_RGB[k]) for k in range(3)], -1)
            mask = np.pad(mask, ((0, ph), (0, pw)), constant_values=IGNORE)
        hh, ww = mask.shape
        for _ in range(10):  # avoid crops dominated by a single class (usually background)
            y = torch.randint(0, hh - c + 1, (1,)).item()
            x = torch.randint(0, ww - c + 1, (1,)).item()
            m = mask[y:y + c, x:x + c]
            counts = np.bincount(m.ravel(), minlength=256)[:NUM_CLASSES]
            if counts.sum() > 0 and np.count_nonzero(counts) > 1 and counts.max() / counts.sum() < self.cat_max_ratio:
                break
        img, mask = img[y:y + c, x:x + c], mask[y:y + c, x:x + c]
        if torch.rand(1).item() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        img = Image.fromarray(np.ascontiguousarray(img))
        if self.jitter is not None:
            img = self.jitter(img)
        img = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)        # uint8 [3,H,W]
        return img, torch.from_numpy(np.ascontiguousarray(mask))               # uint8 [H,W]


class VOCSegTrain(Dataset):
    def __init__(self, split="train", transform=None, root=ROOT, subset=0):
        self.items = read_list(split, root)
        if subset:
            self.items = self.items[:subset]
        self.root, self.transform = root, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(*load_pair(self.root, *self.items[i]))


class InfiniteSampler(Sampler):
    """Endless stream of shuffled epochs, so that training is defined in iterations."""

    def __init__(self, n, seed):
        self.n, self.seed = n, seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.n, generator=g).tolist()


def train_loader(batch_size, crop, seed, workers=6, subset=0, color=True):
    ds = VOCSegTrain("train", TrainTransform(crop, color=color), subset=subset)
    loader = DataLoader(ds, batch_size=batch_size, sampler=InfiniteSampler(len(ds), seed), num_workers=workers,
                        pin_memory=True, drop_last=True, persistent_workers=workers > 0,
                        prefetch_factor=4 if workers else None)
    return ds, loader


class Normalize:
    """uint8 [B,3,H,W] on the GPU -> normalised float tensor."""

    def __init__(self, device="cuda", channels_last=False):
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1) * 255
        self.channels_last = channels_last

    def __call__(self, x):
        x = (x.float() - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last) if self.channels_last else x


# ----------------------------------------------------------------------------- evaluation sets
def boundary_band(mask, width):
    """Pixels closer than `width` px to a label transition (object contour), void excluded."""
    edge = ndimage.maximum_filter(mask, 3, mode="nearest") != ndimage.minimum_filter(mask, 3, mode="nearest")
    return ndimage.binary_dilation(edge, iterations=width) & (mask != IGNORE)


class EvalSet:
    """A whole split decoded into RAM. Images are padded (bottom/right) to 512x512 with the mean
    colour and masks with the void label, so batches are stackable and padding is never scored."""

    def __init__(self, split, root=ROOT, subset=0, band_width=8):
        items = read_list(split, root)
        if subset:
            items = items[:subset]
        self.ids = [i for i, _ in items]
        n, s = len(items), EVAL_SIZE
        self.images = torch.empty((n, 3, s, s), dtype=torch.uint8)
        self.images[:] = torch.tensor(PAD_RGB, dtype=torch.uint8).view(1, 3, 1, 1)
        self.masks = torch.full((n, s, s), IGNORE, dtype=torch.uint8)
        self.band = torch.zeros((n, s, s), dtype=torch.bool)
        self.sizes = np.zeros((n, 2), dtype=np.int64)  # (h, w)

        def load(k):
            img, mask = load_pair(root, *items[k])
            img, mask = np.asarray(img), np.asarray(mask)
            h, w = mask.shape
            self.images[k, :, :h, :w] = torch.from_numpy(img.copy()).permute(2, 0, 1)
            self.masks[k, :h, :w] = torch.from_numpy(mask.copy())
            self.band[k, :h, :w] = torch.from_numpy(boundary_band(mask, band_width))
            self.sizes[k] = (h, w)

        with ThreadPoolExecutor(8) as ex:
            list(ex.map(load, range(n)))

    def __len__(self):
        return len(self.ids)

    def batches(self, batch_size, device="cuda"):
        for i in range(0, len(self), batch_size):
            sl = slice(i, i + batch_size)
            yield (self.images[sl].to(device, non_blocking=True), self.masks[sl].to(device, non_blocking=True),
                   self.band[sl].to(device, non_blocking=True))


def upsample_logits(logits, size):
    return F.interpolate(logits.float(), size=size, mode="bilinear", align_corners=False)

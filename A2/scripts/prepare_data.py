"""Download Pascal VOC 2012 + SBD and build the augmented segmentation set used in A2.

Sources (both official):
  VOC 2012 trainval (2.0 GB)  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  SBD benchmark     (1.4 GB)  https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

Output (data/voc):
  JPEGImages/<id>.jpg          images that have a segmentation label (12,031)
  SegmentationClass/<id>.png   VOC 2012 labels (palette PNG: 0-20 = classes, 255 = void)
  SegmentationClassAug/<id>.png  SBD labels converted from .mat (grayscale PNG, 0-20, no void)
  lists/{train,minival,val}.txt  lines "<id> <label folder>"
  meta.json                      counts and provenance of the split

Split (the usual "train_aug" protocol, plus a held-out monitor set):
  val     = VOC 2012 val (1,449)  -> used as the TEST set, evaluated once per run
  pool    = (VOC12 train + SBD train + SBD val) - VOC12 val = 10,582 images, no overlap with val
  minival = 500 images drawn from pool (seed 0) -> learning curves only
  train   = pool - minival = 10,082

    python -m scripts.prepare_data
"""
import io
import json
import os
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy.io import loadmat

VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
SBD_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
VOC_PREFIX = "VOCdevkit/VOC2012/"
SBD_PREFIX = "benchmark_RELEASE/dataset/"
MINIVAL_SIZE, SPLIT_SEED = 500, 0


def download(url, dst, connections=8):
    """Resumable download with parallel HTTP range requests (the SBD server is slow per connection)."""
    if os.path.isfile(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
        total = int(r.headers["Content-Length"])
    chunk = -(-total // connections)

    def fetch(i):
        start, end = i * chunk, min(total, (i + 1) * chunk) - 1
        part = f"{dst}.part{i}"
        while True:
            have = os.path.getsize(part) if os.path.isfile(part) else 0
            if have >= end - start + 1:
                return part
            req = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                    while block := r.read(1 << 20):
                        f.write(block)
            except OSError as e:
                print(f"  retry part {i}: {e}", flush=True)

    print(f"downloading {url} ({total / 1e9:.2f} GB, {connections} connections)", flush=True)
    with ThreadPoolExecutor(connections) as ex:
        parts = list(ex.map(fetch, range(connections)))
    with open(dst + ".tmp", "wb") as out:
        for part in parts:
            with open(part, "rb") as f:
                while block := f.read(1 << 24):
                    out.write(block)
    assert os.path.getsize(dst + ".tmp") == total
    os.replace(dst + ".tmp", dst)
    for part in parts:
        os.remove(part)


def read_ids(text):
    return [line.strip() for line in text.decode().splitlines() if line.strip()]


def extract_sbd(tgz, out):
    """SBD stores each label as a MATLAB struct; convert GTcls.Segmentation to an index PNG."""
    aug_dir = os.path.join(out, "SegmentationClassAug")
    os.makedirs(aug_dir, exist_ok=True)
    lists, n = {}, 0
    with tarfile.open(tgz, "r:gz") as tar:
        for m in tar:
            if not m.isfile() or not m.name.startswith(SBD_PREFIX):
                continue
            rel = m.name[len(SBD_PREFIX):]
            if rel in ("train.txt", "val.txt"):
                lists[rel[:-4]] = read_ids(tar.extractfile(m).read())
            elif rel.startswith("cls/") and rel.endswith(".mat"):
                mat = loadmat(io.BytesIO(tar.extractfile(m).read()), struct_as_record=False, squeeze_me=True)
                seg = np.asarray(mat["GTcls"].Segmentation, dtype=np.uint8)
                Image.fromarray(seg).save(os.path.join(aug_dir, os.path.basename(rel)[:-4] + ".png"))
                n += 1
                if n % 2000 == 0:
                    print(f"  SBD labels converted: {n}", flush=True)
    return lists, n


def extract_voc(tar_path, out, wanted_jpegs):
    lists = {}
    for d in ("JPEGImages", "SegmentationClass"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    with tarfile.open(tar_path, "r:") as tar:
        for m in tar:
            if not m.isfile() or not m.name.startswith(VOC_PREFIX):
                continue
            rel = m.name[len(VOC_PREFIX):]
            if rel.startswith("ImageSets/Segmentation/") and rel.endswith(".txt"):
                lists[os.path.basename(rel)[:-4]] = read_ids(tar.extractfile(m).read())
            elif rel.startswith("SegmentationClass/") or (
                    rel.startswith("JPEGImages/") and os.path.basename(rel)[:-4] in wanted_jpegs):
                with open(os.path.join(out, rel), "wb") as f:
                    f.write(tar.extractfile(m).read())
    return lists


if __name__ == "__main__":
    root, out = "data/downloads", "data/voc"
    voc_tar, sbd_tgz = os.path.join(root, os.path.basename(VOC_URL)), os.path.join(root, os.path.basename(SBD_URL))
    download(VOC_URL, voc_tar, connections=4)
    download(SBD_URL, sbd_tgz, connections=8)

    print("extracting SBD labels ...", flush=True)
    sbd, n_sbd = extract_sbd(sbd_tgz, out)
    sbd_ids = set(sbd["train"]) | set(sbd["val"])
    print("extracting VOC 2012 ...", flush=True)
    # JPEG names are only known after reading the VOC lists, so take every image that SBD or VOC labels
    with tarfile.open(voc_tar, "r:") as tar:
        voc_seg_ids = {os.path.basename(m.name)[:-4] for m in tar
                       if m.name.startswith(VOC_PREFIX + "SegmentationClass/") and m.isfile()}
    voc = extract_voc(voc_tar, out, sbd_ids | voc_seg_ids)

    val = sorted(voc["val"])
    pool = sorted((set(voc["train"]) | sbd_ids) - set(val))
    rng = np.random.RandomState(SPLIT_SEED)
    minival = sorted(rng.choice(pool, MINIVAL_SIZE, replace=False).tolist())
    train = sorted(set(pool) - set(minival))
    assert not set(train) & set(val) and not set(minival) & set(val) and not set(train) & set(minival)

    os.makedirs(os.path.join(out, "lists"), exist_ok=True)
    label_dir = lambda i: "SegmentationClass" if i in voc_seg_ids else "SegmentationClassAug"  # noqa: E731
    for name, ids in (("train", train), ("minival", minival), ("val", val)):
        for i in ids:
            assert os.path.isfile(os.path.join(out, "JPEGImages", i + ".jpg")), i
            assert os.path.isfile(os.path.join(out, label_dir(i), i + ".png")), i
        with open(os.path.join(out, "lists", name + ".txt"), "w") as f:
            f.writelines(f"{i} {label_dir(i)}\n" for i in ids)
    meta = {
        "voc12_train": len(voc["train"]), "voc12_val": len(voc["val"]), "voc12_seg_labels": len(voc_seg_ids),
        "sbd_train": len(sbd["train"]), "sbd_val": len(sbd["val"]), "sbd_labels": n_sbd,
        "sbd_overlap_voc12_val": len(sbd_ids & set(val)), "pool_train_aug": len(pool),
        "train": len(train), "minival": len(minival), "val": len(val),
        "train_voc_labels": sum(i in voc_seg_ids for i in train), "split_seed": SPLIT_SEED,
        "sources": {"voc2012": VOC_URL, "sbd": SBD_URL},
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))

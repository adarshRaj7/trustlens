"""
Generates labeled splice/authentic patches from royalty-free photos you supply.

Usage:
  1. Drop 4+ diverse photos (JPEG) into training/sources/
  2. python build_dataset.py

No public splice-detection dataset exists that's freely redistributable, so
this synthesizes labeled examples: a patch from one photo is pre-compressed,
then pasted into another and the composite re-saved -- reproducing a genuine
double-JPEG-compression fingerprint, the same signal a real splice leaves.
"""

import glob
import io
import os
import random

from PIL import Image

random.seed(42)

SOURCES_DIR = os.path.join(os.path.dirname(__file__), "sources")
W, H = 1024, 768
PATCH = 128


def load_sources():
    paths = sorted(glob.glob(os.path.join(SOURCES_DIR, "*.jpg")) +
                    glob.glob(os.path.join(SOURCES_DIR, "*.jpeg")))
    if len(paths) < 2:
        raise RuntimeError(
            f"Need at least 2 photos in {SOURCES_DIR} -- found {len(paths)}. "
            "Drop in royalty-free JPEGs and rerun."
        )
    imgs = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        imgs[name] = Image.open(p).convert("RGB").resize((W, H))
    return imgs


def jpeg_roundtrip(im, quality):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rand_box(size=PATCH):
    x = random.randint(0, W - size)
    y = random.randint(0, H - size)
    return x, y, x + size, y + size


def make_authentic_patch(imgs, source_name):
    base = imgs[source_name]
    q_final = random.randint(70, 95)
    composite = jpeg_roundtrip(base, q_final)
    box = rand_box()
    patch = composite.crop(box)
    return patch, frozenset([source_name])


def make_spliced_patch(imgs, host_name, donor_name):
    host = imgs[host_name]
    donor = imgs[donor_name]
    q_donor = random.randint(50, 90)
    q_final = random.randint(70, 95)
    donor_compressed = jpeg_roundtrip(donor, q_donor)

    size = PATCH
    dx, dy, dx2, dy2 = rand_box(size)
    donor_patch = donor_compressed.crop((dx, dy, dx2, dy2))

    hx, hy, hx2, hy2 = rand_box(size)
    composite = host.copy()
    composite.paste(donor_patch, (hx, hy))
    composite_final = jpeg_roundtrip(composite, q_final)

    tampered_patch = composite_final.crop((hx, hy, hx2, hy2))
    return tampered_patch, frozenset([host_name, donor_name])


def build(n_per_class=400, imgs=None):
    imgs = imgs or load_sources()
    names = list(imgs.keys())
    if len(names) < 2:
        raise RuntimeError("Need at least 2 source photos to build splice pairs.")
    records = []
    for _ in range(n_per_class):
        src = random.choice(names)
        patch, group = make_authentic_patch(imgs, src)
        records.append((patch, 0, group))
    for _ in range(n_per_class):
        host, donor = random.sample(names, 2)
        patch, group = make_spliced_patch(imgs, host, donor)
        records.append((patch, 1, group))
    random.shuffle(records)
    return records


if __name__ == "__main__":
    imgs = load_sources()
    print(f"Loaded {len(imgs)} source photos: {list(imgs.keys())}")
    records = build(400, imgs)
    print(f"Built {len(records)} patches: "
          f"{sum(1 for r in records if r[1] == 0)} authentic, "
          f"{sum(1 for r in records if r[1] == 1)} spliced")

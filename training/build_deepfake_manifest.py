"""
Downloads (via kagglehub) and merges the three deepfake face datasets into
one train/val/test manifest, with content-hash deduplication so the same
real FFHQ photo (which multiple Kaggle mirrors reuse) can't leak across
splits, and per-source-dataset labels preserved so training/eval can
report accuracy per generator family (StyleGAN1 / StyleGAN3 / diffusion)
instead of one aggregate number that the largest dataset would dominate.

Kaggle auth (required, do this on the training machine -- never paste an
API key into a chat session): download a token from
https://www.kaggle.com/settings -> "Create New Token", save it as
~/.kaggle/kaggle.json, then `chmod 600 ~/.kaggle/kaggle.json`. Or set the
KAGGLE_USERNAME / KAGGLE_KEY environment variables instead.

Labeling: images are auto-labeled real/fake from folder-name keywords
("real"/"authentic" -> 0, "fake"/"gan"/"synthetic"/"generated"/"ai" -> 1).
A dataset that contains ONLY generated images (no matching "real" folder --
e.g. a pure Stable-Diffusion-faces dump) must be passed with --fake-only;
every image in it is labeled fake and the "real" class is pooled from the
other datasets, since a real photo is a real photo regardless of which
mirror it came from.

Usage (default handles resolve automatically -- see DEFAULT_HANDLES below;
override with name=owner/slug for a different Kaggle dataset, or
name=/local/path if you already have it on disk):
  python build_deepfake_manifest.py \
    --dataset stylegan1 --dataset stylegan3 \
    --dataset diffusion --fake-only diffusion \
    --cache-dir /path/on/a/partition/with/room \
    --out manifest.csv

--cache-dir matters: kagglehub defaults to caching under your home
directory, which may not have room (ours didn't -- check `df -h` first).
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from PIL import Image, UnidentifiedImageError

try:
    import imagehash
    HAVE_IMAGEHASH = True
except ImportError:
    HAVE_IMAGEHASH = False

IMG_EXTS = {".jpg", ".jpeg", ".png"}
REAL_KEYWORDS = {"real", "authentic"}
FAKE_KEYWORDS = {"fake", "gan", "synthetic", "generated", "ai"}

DEFAULT_HANDLES = {
    "stylegan1": "manjilkarki/deepfake-and-real-images",
    "stylegan3": "troykueh/real-vs-fake-faces-stylegan3",
    "diffusion": "mohannadaymansalah/stable-diffusion-dataaaaaaaaa",
}

KAGGLE_HANDLE_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def check_kaggle_auth():
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    sys.exit(
        "No Kaggle API credentials found. Either:\n"
        "  1. Download a token from https://www.kaggle.com/settings -> "
        "'Create New Token', save as ~/.kaggle/kaggle.json, then: "
        "chmod 600 ~/.kaggle/kaggle.json\n"
        "  2. Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
        "Set this up directly on the training machine -- never paste an API key into a chat session."
    )


def resolve_root(name: str, spec: str | None) -> Path:
    """spec is a local path, a Kaggle 'owner/slug' handle, or None (use
    DEFAULT_HANDLES[name]). Downloads via kagglehub if it's a handle."""
    if spec is None:
        spec = DEFAULT_HANDLES.get(name)
        if spec is None:
            sys.exit(f"No default Kaggle handle for '{name}' -- pass "
                      f"--dataset {name}=owner/slug or --dataset {name}=/local/path")

    local = Path(spec)
    if local.exists():
        return local

    if KAGGLE_HANDLE_RE.match(spec):
        import kagglehub
        print(f"[{name}] downloading Kaggle dataset '{spec}' via kagglehub...")
        return Path(kagglehub.dataset_download(spec))

    sys.exit(f"'{spec}' for dataset '{name}' is neither an existing local "
              f"path nor a valid Kaggle handle (owner/slug)")


def infer_label(rel_path: Path, force_fake: bool):
    """Labels from the path BELOW the dataset root, innermost component first.

    Three things this has to get right, each learned from a real dataset:

    - Components are word-split ("Fake faces" -> {"fake", "faces"}) rather than
      compared whole, because datasets name folders "Real faces", not "real".
    - Only components below the dataset root are considered. The Kaggle dataset
      directory is itself often named "real-vs-fake-faces-stylegan3" or
      "deepfake-and-real-images", which word-splits to contain "real" -- match
      against the full path and EVERY image in the dataset silently takes
      whichever class is tested first, Fake/ folder contents included.
    - Innermost component wins, so a "Fake" leaf folder beats a "real_vs_fake"
      parent instead of losing to whichever keyword happens to be checked first.

    Substring matching would be looser still: "ai" is a substring of plenty of
    unrelated words ("training", "explain"), hence whole-word matching.
    """
    if force_fake:
        return 1
    for part in reversed(rel_path.parts):
        words = set(re.split(r"[^a-z0-9]+", part.lower()))
        if words & REAL_KEYWORDS:
            return 0
        if words & FAKE_KEYWORDS:
            return 1
    return None


def content_key(path: Path) -> str:
    """Dedup key: perceptual hash if available (catches the same photo
    re-compressed by a different dataset mirror), else raw byte hash.

    DCT-based phash, not average_hash: this corpus is aligned, centered face
    crops that all share one luminance layout, and aHash's 8x8 mean-threshold
    signature collides across genuinely different people on exactly that kind
    of homogeneous data. Since dedup keeps the first of each key, a collision
    silently deletes a distinct face rather than a duplicate one.
    """
    if HAVE_IMAGEHASH:
        try:
            return "phash:" + str(imagehash.phash(Image.open(path)))
        except Exception:
            pass
    return "md5:" + hashlib.md5(path.read_bytes()[:65536]).hexdigest()


def split_for(key: str) -> str:
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def is_readable(path: Path) -> bool:
    """Large real-world Kaggle dumps reliably contain a handful of
    truncated/corrupt files. Catch them here, once, rather than crashing
    a training run hours in when DataLoader hits one."""
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def scan_dataset(name: str, root: Path, force_fake: bool):
    rows = []
    skipped_unlabeled = 0
    skipped_corrupt = 0
    for path in root.rglob("*"):
        if path.suffix.lower() not in IMG_EXTS:
            continue
        label = infer_label(path.relative_to(root), force_fake)
        if label is None:
            skipped_unlabeled += 1
            continue
        if not is_readable(path):
            skipped_corrupt += 1
            continue
        rows.append({"path": str(path), "source": name, "label": label})
    if skipped_unlabeled:
        print(f"[{name}] skipped {skipped_unlabeled} files with no real/fake keyword in path")
    if skipped_corrupt:
        print(f"[{name}] skipped {skipped_corrupt} unreadable/corrupt image files")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", required=True,
                     help="NAME, or NAME=owner/kaggle-slug, or NAME=/local/path -- repeatable. "
                          f"Bare NAME resolves via DEFAULT_HANDLES: {list(DEFAULT_HANDLES)}")
    ap.add_argument("--fake-only", action="append", default=[],
                     help="dataset name(s) that contain only generated images")
    ap.add_argument("--cache-dir", default=None,
                     help="where kagglehub stores downloads -- put this on a partition "
                          "with real free space, not wherever your home directory sits")
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()

    check_kaggle_auth()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
        free_gb = shutil.disk_usage(cache_dir).free / 1e9
        if free_gb < 20:
            print(f"[warn] only {free_gb:.1f}GB free at {cache_dir} -- these three "
                  f"datasets combined are well over that; downloads may fail partway through")

    if not HAVE_IMAGEHASH:
        print("[warn] imagehash not installed -- dedup falls back to exact byte "
              "hash, which WON'T catch the same photo re-compressed by a "
              "different dataset mirror. `pip install imagehash` for a stronger "
              "dedup pass before training.")

    fake_only = set(args.fake_only)
    rows = []
    for spec in args.dataset:
        name, _, val = spec.partition("=")
        root = resolve_root(name, val or None)
        rows.extend(scan_dataset(name, root, force_fake=name in fake_only))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No labeled images found -- check --dataset paths and folder naming.")

    print("Hashing for dedup + split assignment (walks every file once)...")
    df["content_key"] = [content_key(Path(p)) for p in df["path"]]
    before = len(df)
    df = df.drop_duplicates(subset="content_key", keep="first")
    print(f"Deduped {before - len(df)} near/exact-duplicate images ({before} -> {len(df)})")

    df["split"] = df["content_key"].map(split_for)

    print("\nPer source / label / split counts:")
    print(df.groupby(["source", "label", "split"]).size())

    df[["path", "label", "source", "split"]].to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

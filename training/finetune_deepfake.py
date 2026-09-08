"""
Fine-tunes the EfficientNet-B4 deepfake classifier used by
backend/models/deepfake_model.py, on the merged real/fake face manifest
from build_deepfake_manifest.py (StyleGAN1 + StyleGAN3 + diffusion fakes,
pooled real photos).

Two-phase transfer learning: classifier head only for --freeze-epochs,
then the whole network unfrozen at a lower LR. Validation is reported BOTH
pooled and broken out per source dataset -- the pooled number alone would
be dominated by whichever dataset is largest (currently the 140k StyleGAN1
set), which would hide whether the model actually generalizes to the other
generator families. Checkpoint selection uses the mean of per-source val
AUCs, not pooled accuracy, so the largest source can't dominate selection.

Input resolution is 380x380 to match EfficientNet-B4's native training
scale. backend/models/deepfake_model.py's inference transform is kept at
the same resolution -- if you change IMG_SIZE here, update it there too.

Usage:
  python build_deepfake_manifest.py --dataset ... --out manifest.csv
  python finetune_deepfake.py --manifest manifest.csv --hf-repo you/trustlens-deepfake-b4

--hf-repo uploads the best checkpoint + model card to Hugging Face Hub once
training finishes (needs `huggingface-cli login` or HF_TOKEN set on this
machine first -- never paste a token into a chat session). This is the
recommended way to get the trained weights off a compute node whose local
disk isn't persistent -- see backend/models/deepfake_model.py, which will
auto-download from the same repo if no local weights file is present.
"""

import argparse
import os

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

IMG_SIZE = 380  # EfficientNet-B4 native resolution -- must match deepfake_model.py
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "weights", "efficientnet_b4_ff.pth")
CARD_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "weights", "DEEPFAKE_MODEL_CARD.md")

NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class FaceDataset(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        # label convention matches deepfake_model.py: class 1 = real
        y = 1 - int(row["label"])
        return self.tf(img), y


def make_sampler(df):
    """Equal expected mass per source dataset, so the largest dataset
    doesn't drown out the smaller StyleGAN3/diffusion ones during training."""
    counts = df["source"].value_counts()
    weights = df["source"].map(lambda s: 1.0 / counts[s]).to_numpy()
    return WeightedRandomSampler(weights, num_samples=len(df), replacement=True)


def evaluate(model, df, device, batch_size, num_workers):
    model.eval()
    ds = FaceDataset(df, eval_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=(device == "cuda"))
    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                probs = torch.softmax(model(x), dim=1)[:, 1].float().cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.numpy())
    probs = np.concatenate(all_probs)
    y = np.concatenate(all_y)
    df = df.reset_index(drop=True)

    results = {}
    for source in df["source"].unique():
        mask = (df["source"] == source).to_numpy()
        try:
            results[source] = roc_auc_score(y[mask], probs[mask])
        except ValueError:
            results[source] = float("nan")
    results["_pooled"] = roc_auc_score(y, probs)
    results["_mean_per_source"] = float(
        np.nanmean([v for k, v in results.items() if not k.startswith("_")])
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-full", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hf-repo", default=None,
                     help="e.g. you/trustlens-deepfake-b4 -- if set, uploads the best "
                          "checkpoint + model card to HF Hub after training")
    ap.add_argument("--hf-private", action="store_true",
                     help="create the HF repo as private (default: public)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.manifest)
    train_df = df[df.split == "train"]
    val_df = df[df.split == "val"]
    test_df = df[df.split == "test"]
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("Train sources:\n", train_df["source"].value_counts())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("[warn] no CUDA device found -- this will be very slow for "
              "EfficientNet-B4 at 380x380. Check the GPU server setup first.")

    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()

    train_ds = FaceDataset(train_df, train_tf)
    sampler = make_sampler(train_df)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=(device == "cuda"))

    best_score = -1.0
    opt = None
    current_phase = None

    for epoch in range(args.epochs):
        frozen = epoch < args.freeze_epochs
        phase = "head" if frozen else "full"
        if phase != current_phase:
            # only rebuild the optimizer at a phase transition -- rebuilding it
            # every epoch would reset AdamW's running gradient/variance
            # estimates each time, which hurts convergence during the (longer,
            # more sensitive) full fine-tune phase
            for p in model.parameters():
                p.requires_grad = not frozen
            for p in model.get_classifier().parameters():
                p.requires_grad = True
            lr = args.lr_head if frozen else args.lr_full
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
            current_phase = phase

        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch} ({phase})"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        val_report = evaluate(model, val_df, device, args.batch_size, args.num_workers)
        print(f"epoch {epoch}: train_loss={total_loss/len(train_ds):.4f} val={val_report}")

        if val_report["_mean_per_source"] > best_score:
            best_score = val_report["_mean_per_source"]
            torch.save(model.state_dict(), OUT_PATH)
            print(f"  -> new best (mean per-source AUC={best_score:.3f}), saved to {OUT_PATH}")

    if best_score < 0:
        # every epoch's mean-per-source AUC was NaN, so the comparison above
        # (NaN > x is always False) never saved anything. Fail loudly here
        # rather than at the torch.load below, whose FileNotFoundError says
        # nothing about the real cause: a val split where no source has both
        # classes, i.e. a mislabeled manifest. Check the per-source/label
        # counts printed by build_deepfake_manifest.py before re-running.
        raise RuntimeError(
            "No checkpoint was saved -- mean per-source validation AUC was NaN "
            "every epoch, meaning no source had both real and fake images in "
            "the val split. Rebuild the manifest and check its per-source/"
            "label/split counts before retraining."
        )

    # final honest report on held-out test split, using the BEST checkpoint
    model.load_state_dict(torch.load(OUT_PATH, map_location=device))
    test_report = evaluate(model, test_df, device, args.batch_size, args.num_workers)
    print("\nFinal test-set report (best checkpoint):", test_report)

    with open(CARD_PATH, "w") as f:
        f.write("# Deepfake Model Card\n\n")
        f.write("EfficientNet-B4, fine-tuned on a merged real/fake face manifest:\n")
        f.write("StyleGAN1 (manjilkarki/deepfake-and-real-images), "
                "StyleGAN3 (troykueh/real-vs-fake-faces-stylegan3), "
                "diffusion (mohannadaymansalah/stable-diffusion-dataaaaaaaaa).\n\n")
        f.write("Real photos are pooled across sources (content-hash deduped); "
                "fakes are kept per-generator so cross-generator generalization "
                "can be measured directly instead of inferred from pooled accuracy.\n\n")
        f.write("## Held-out test AUC, by generator family\n\n")
        for k, v in test_report.items():
            f.write(f"- {k}: {v:.3f}\n")
        f.write("\nSelection criterion during training: best mean-per-source "
                "validation AUC (not pooled accuracy), so the largest source "
                "dataset can't dominate checkpoint selection.\n")
    print(f"Wrote {CARD_PATH}")

    if args.hf_repo:
        upload_to_hf(args.hf_repo, args.hf_private)


def upload_to_hf(repo_id: str, private: bool):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[warn] huggingface_hub not installed -- skipping upload. "
              "`pip install huggingface_hub` and re-run with --hf-repo to push weights.")
        return

    try:
        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        api.upload_file(path_or_fileobj=OUT_PATH, path_in_repo="efficientnet_b4_ff.pth", repo_id=repo_id)
        # CARD_PATH doubles as the HF model card -- HF renders README.md on the repo page
        api.upload_file(path_or_fileobj=CARD_PATH, path_in_repo="README.md", repo_id=repo_id)
        print(f"Uploaded checkpoint + model card to https://huggingface.co/{repo_id} "
              f"({'private' if private else 'public'})")
    except Exception as e:
        # the checkpoint is already saved locally at this point -- an upload
        # failure shouldn't read as "training failed"
        print(f"[warn] HF Hub upload failed: {e}\n"
              f"The trained checkpoint is still safe at {OUT_PATH} -- "
              "check `huggingface-cli login` / HF_TOKEN and retry the upload manually if needed.")


if __name__ == "__main__":
    main()

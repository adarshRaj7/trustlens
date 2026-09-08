# Splice Classifier — Model Card

## Problem

No public image-splice dataset (e.g. CASIA v2) was available for this project.
Rather than ship an untrained classifier or a `random.random()` mock, this
model is trained on **synthetically generated** splice examples.

## Data generation

Source material: 4 royalty-free stock photos (landscape/desert/macbook wallpaper).
For each labeled patch (128x128px):

- **Authentic**: the whole source photo is JPEG-compressed once at a random
  quality (70-95), then a patch is cropped from it.
- **Spliced**: a patch is cropped from one photo (the "donor"), which is
  itself pre-compressed at a random quality (50-90) *before* being pasted into
  a different "host" photo; the composite is then saved at a final random
  quality (70-95). This reproduces a genuine double-JPEG-compression
  fingerprint, the same signal a real splice leaves behind.

800 patches total (400 authentic / 400 spliced) in the shipped model.

## Features (see `splice_features.py`)

Error Level Analysis stats, JPEG Ghost best-fit quality + flatness, noise
residual variance, FFT high-frequency energy ratio — 7 features total.

## Model

`StandardScaler` + `RandomForestClassifier` (n_estimators=300, max_depth=5,
min_samples_leaf=5), via scikit-learn.

## Validation — read this before quoting a number

**Random 5-fold CV: 71.7% accuracy, F1 0.736.** This number is optimistic:
patches from the same source photo appear in both the train and test folds,
so the model can partly key off scene-specific content rather than general
splice signal.

**Leave-one-source-photo-out (the honest test — train on 3 photos, test only
on patches involving the 4th, never seen in training):**

| Held-out photo | Accuracy | F1 | AUC |
|---|---|---|---|
| mountain | 0.59 | 0.57 | 0.87 |
| desert | 0.59 | 0.70 | 0.55 |
| unsplash | 0.58 | 0.69 | 0.50 |
| macbook | 0.60 | 0.60 | 0.88 |
| **mean** | **0.59** | **0.64** | **0.70** |

Interpretation: real, above-chance signal, but it does not reliably
generalize yet — AUC swings from 0.50 (no better than a coin flip) to 0.88
depending on which photo is held out. With only 4 source images, that
variance is expected sample noise, not evidence the features are useless.

## The honest path to a stronger number

This is a data problem, not a modeling problem: rerun `build_dataset.py` and
`train_eval.py` with dozens-to-hundreds of diverse stock photos instead of 4,
and re-check the leave-one-photo-out AUC. Do this before quoting a
higher accuracy number anywhere, including in an interview.

## What NOT to claim

- Do not claim this detects deepfakes (face-swap/GAN-generated faces) — it
  targets JPEG splice/composite forgery only, a different problem.
- Do not quote the 71.7% random-split number as the model's real performance.
- Do not claim CASIA-v2-benchmark-comparable accuracy — this was never
  evaluated against that or any standard forensics benchmark.

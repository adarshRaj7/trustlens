"""
Trains the splice classifier and validates it honestly.

Usage: python train_eval.py

Prints TWO numbers -- use the second one, not the first:
  1. Random 5-fold CV accuracy/F1 -- optimistic, source photos leak across folds.
  2. Leave-one-source-photo-out accuracy/F1/AUC -- the real generalization test
     (train on all photos but one, test only on patches touching the held-out
     photo). This is what belongs on a resume/in an interview.

Saves the final model (trained on all data) to ../backend/weights/splice_classifier.pkl
"""

import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend", "models"))
from splice_features import extract_features, FEATURE_NAMES  # noqa: E402

from build_dataset import build, load_sources  # noqa: E402

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "weights", "splice_classifier.pkl")


def make_clf():
    return make_pipeline(StandardScaler(), RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42))


def main():
    imgs = load_sources()
    names = list(imgs.keys())
    print(f"Source photos ({len(names)}): {names}")

    records = build(n_per_class=400, imgs=imgs)
    X = np.array([extract_features(patch) for patch, _, _ in records])
    y = np.array([label for _, label, _ in records])
    groups = [group for _, _, group in records]
    print(f"Dataset: {X.shape[0]} patches, {X.shape[1]} features")

    # 1. optimistic random-split CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        clf = make_clf()
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred))
    print(f"\n[Random 5-fold CV -- OPTIMISTIC, do not quote this alone]")
    print(f"  Accuracy: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"  F1:       {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")

    # 2. honest leave-one-photo-out
    print(f"\n[Leave-one-source-photo-out -- the honest generalization number]")
    aucs = []
    for held_out in names:
        train_mask = np.array([held_out not in g for g in groups])
        test_mask = ~train_mask
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        clf = make_clf()
        clf.fit(X[train_mask], y[train_mask])
        pred = clf.predict(X[test_mask])
        proba = clf.predict_proba(X[test_mask])[:, 1]
        acc = accuracy_score(y[test_mask], pred)
        prec = precision_score(y[test_mask], pred, zero_division=0)
        rec = recall_score(y[test_mask], pred, zero_division=0)
        f1 = f1_score(y[test_mask], pred, zero_division=0)
        try:
            auc = roc_auc_score(y[test_mask], proba)
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        print(f"  held out '{held_out}': n={test_mask.sum():4d} acc={acc:.3f} "
              f"prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} auc={auc:.3f}")
    print(f"\n  Mean AUC across held-out photos: {np.nanmean(aucs):.3f}  <-- use this number")

    # feature importances
    clf_full = make_clf()
    clf_full.fit(X, y)
    importances = clf_full.named_steps["randomforestclassifier"].feature_importances_
    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda t: -t[1]):
        print(f"  {name:16s} {imp:.3f}")

    joblib.dump({"pipeline": clf_full, "feature_names": FEATURE_NAMES}, OUT_PATH)
    print(f"\nSaved shipped model to {OUT_PATH}")
    print("Now update backend/weights/SPLICE_MODEL_CARD.md with the numbers printed above.")


if __name__ == "__main__":
    main()

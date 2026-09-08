"""
Image Tampering Detection Model
────────────────────────────────
Primary detector: a Random Forest trained on hand-crafted forensic features
(ELA, JPEG Ghost, noise residual, FFT high-frequency ratio) — see
splice_features.py. No public splice-detection dataset was available, so the
training data is synthetically generated: patches are spliced from one stock
photo into another with an intermediate JPEG re-compression step (creating a
genuine double-JPEG-compression fingerprint), then labeled authentic/spliced.

Validated with a leave-one-source-photo-out protocol (never train and test on
patches from the same source image) — mean AUC 0.70, ranging 0.50-0.88 depending
on the held-out photo. This is real, moderate, and honestly reported: full
methodology and numbers in weights/SPLICE_MODEL_CARD.md. It is NOT a claim of
CASIA-v2-benchmark-level accuracy.

Secondary/experimental: a ViT-B/16 backbone loads real ImageNet-pretrained
weights, but its classification head is untrained (random init) — kept here
for future fine-tuning once a labeled dataset (e.g. CASIA v2) is available,
NOT used in predict() by default because its output is not currently meaningful.
"""

import hashlib
import random
import io
from pathlib import Path

import joblib
from PIL import Image

from .splice_features import extract_features

CLASSIFIER_PATH = Path(__file__).parent.parent / "weights" / "splice_classifier.pkl"
VIT_WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "vit_tamper_casia.pth"


class TamperingDetector:
    """
    Tamper detector. predict() returns score 0-100 where 100 = predicted authentic.
    """

    def __init__(self):
        self.classifier = None
        self.feature_names = None
        self._load_classifier()

        # experimental ViT path, loaded but not used by predict() -- see module docstring
        self.vit_model = None
        self.vit_processor = None
        self._load_vit()

    def _load_classifier(self):
        if CLASSIFIER_PATH.exists():
            bundle = joblib.load(CLASSIFIER_PATH)
            self.classifier = bundle["pipeline"]
            self.feature_names = bundle["feature_names"]
            print("[TamperingDetector] Loaded trained splice classifier "
                  "(synthetic-splice data, leave-one-photo-out mean AUC 0.70)")
        else:
            print("[TamperingDetector] No trained classifier found at "
                  f"{CLASSIFIER_PATH} — falling back to mock mode")

    def _load_vit(self):
        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            import torch

            model_name = "google/vit-base-patch16-224"
            self.vit_processor = ViTImageProcessor.from_pretrained(model_name)
            self.vit_model = ViTForImageClassification.from_pretrained(
                model_name, num_labels=2, ignore_mismatched_sizes=True
            )
            if VIT_WEIGHTS_PATH.exists():
                state = torch.load(VIT_WEIGHTS_PATH, map_location="cpu")
                self.vit_model.load_state_dict(state)
                print("[TamperingDetector] (experimental) Loaded CASIA fine-tuned ViT weights")
            else:
                print("[TamperingDetector] (experimental) ViT backbone loaded, "
                      "classification head is UNTRAINED — not used by predict()")
            self.vit_model.eval()
        except ImportError:
            pass  # experimental path is optional

    PATCH_SIZE = 128  # must match training patch size in build_dataset.py

    def predict(self, image_bytes: bytes) -> int:
        """
        Returns authenticity score 0-100 (100 = predicted untampered).
        The classifier was trained on 128x128 patches, so we slide over
        patches at that same scale. We score on the mean of the top-3
        spliced-probabilities rather than a single hard max: a genuine
        splice is local, so averaging over the whole image would dilute it,
        but a plain max over dozens of patches all but guarantees at least
        one false-positive-looking patch by chance on an ordinary photo.
        Requiring a few corroborating patches keeps local sensitivity while
        cutting single-outlier noise.
        """
        if self.classifier is None:
            return self._mock_predict(image_bytes)

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            probas = sorted((p for _, p in self._scan_patches(img)), reverse=True)
            top = probas[:3] if probas else [0.0]
            spliced_proba = sum(top) / len(top)
            return int(round((1 - spliced_proba) * 100))
        except Exception as e:
            print(f"[TamperingDetector] Error: {e}")
            return self._mock_predict(image_bytes)

    def _scan_patches(self, img):
        """Yields ((grid_y, grid_x), spliced_probability) for non-overlapping
        PATCH_SIZE tiles -- the same scale the classifier was trained on."""
        ps = self.PATCH_SIZE
        w, h = img.size
        if w < ps or h < ps:
            img = img.resize((max(w, ps), max(h, ps)))
            w, h = img.size
        for gy, y in enumerate(range(0, h - ps + 1, ps)):
            for gx, x in enumerate(range(0, w - ps + 1, ps)):
                patch = img.crop((x, y, x + ps, y + ps))
                feats = [extract_features(patch)]
                proba_spliced = self.classifier.predict_proba(feats)[0][1]
                yield (gy, gx), proba_spliced

    def predict_heatmap(self, image_bytes: bytes):
        """
        Blockwise authenticity map: slides the trained classifier over
        PATCH_SIZE tiles so the frontend can highlight which regions look spliced.
        """
        if self.classifier is None:
            return self._mock_heatmap()

        try:
            import numpy as np
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            cells = list(self._scan_patches(img))
            n_gy = max(gy for (gy, gx), _ in cells) + 1
            n_gx = max(gx for (gy, gx), _ in cells) + 1
            heat = np.zeros((n_gy, n_gx), dtype=np.float32)
            for (gy, gx), proba in cells:
                heat[gy, gx] = proba
            return heat
        except Exception as e:
            print(f"[TamperingDetector] Heatmap error: {e}")
            return self._mock_heatmap()

    def _mock_predict(self, image_bytes: bytes) -> int:
        h = int(hashlib.md5(image_bytes[:256]).hexdigest(), 16)
        return random.Random(h + 1).randint(15, 90)

    def _mock_heatmap(self):
        import numpy as np
        return np.random.rand(6, 6)

"""
Deepfake Detection Model
─────────────────────────
Wraps EfficientNet-B4, fine-tuned by training/finetune_deepfake.py on a
merged real/fake face dataset (see that script and
backend/weights/DEEPFAKE_MODEL_CARD.md). Falls back to a neutral score if
no fine-tuned weights are available anywhere.

Weight resolution order:
  1. Local file at backend/weights/efficientnet_b4_ff.pth
  2. Hugging Face Hub repo named by the TRUSTLENS_DEEPFAKE_HF_REPO env var
     (e.g. "you/trustlens-deepfake-b4") -- this is how a training run on a
     machine with no persistent local disk gets its weights into a deployed
     backend, via `finetune_deepfake.py --hf-repo ...`.
  3. Neither found -> NEUTRAL_SCORE, no inference run.
"""

import hashlib
import os
import random
from pathlib import Path
from typing import Optional


WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "efficientnet_b4_ff.pth"
HF_REPO_ENV = "TRUSTLENS_DEEPFAKE_HF_REPO"


class DeepfakeDetector:
    """
    Deepfake detector using EfficientNet-B4.
    Returns a score 0-100 where 100 = definitely real.
    """

    # Neutral score returned when no fine-tuned head is available: the
    # ImageNet backbone carries no deepfake signal, and a freshly
    # initialized 2-class head is pure noise, so we do not run inference
    # through it (its output would sometimes be < 20, wrongly tripping the
    # fusion engine's "confident fake" cap on ordinary real photos).
    NEUTRAL_SCORE = 75

    def __init__(self):
        self.model = None
        self.transform = None
        self.finetuned = False
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import timm
            from torchvision import transforms

            self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=2)

            weights_path = WEIGHTS_PATH if WEIGHTS_PATH.exists() else self._fetch_from_hf()

            if weights_path:
                state = torch.load(weights_path, map_location='cpu')
                self.model.load_state_dict(state)
                self.finetuned = True
                print(f"[DeepfakeDetector] ✓ Loaded fine-tuned weights ({weights_path})")
            else:
                print("[DeepfakeDetector] ⚠ No fine-tuned weights — classifier head is "
                      "untrained noise, predict() will return a neutral score instead "
                      "of running inference")

            self.model.eval()
            # 380x380 = EfficientNet-B4's native resolution, and what
            # training/finetune_deepfake.py trains at -- must stay in sync
            # with that script's IMG_SIZE, or a loaded checkpoint sees a
            # different input distribution than it was trained on.
            self.transform = transforms.Compose([
                transforms.Resize((380, 380)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        except ImportError:
            print("[DeepfakeDetector] ⚠ timm/torch not installed — mock mode")

    def _fetch_from_hf(self) -> Optional[Path]:
        """Downloads the checkpoint from HF Hub if TRUSTLENS_DEEPFAKE_HF_REPO
        is set. Returns None (and falls back to NEUTRAL_SCORE upstream) on
        any failure -- a missing/misconfigured repo shouldn't crash startup."""
        repo_id = os.environ.get(HF_REPO_ENV)
        if not repo_id:
            return None
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo_id, filename="efficientnet_b4_ff.pth")
            return Path(path)
        except Exception as e:
            print(f"[DeepfakeDetector] ⚠ Could not fetch weights from HF Hub "
                  f"repo '{repo_id}': {e}")
            return None

    def predict(self, image_bytes: bytes) -> int:
        """Returns authenticity score 0-100 (100 = real)."""
        if self.model is None:
            return self._mock_predict(image_bytes)

        if not self.finetuned:
            return self.NEUTRAL_SCORE

        try:
            import torch
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            tensor = self.transform(img).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                real_prob = probs[0][1].item()  # class 1 = real

            return int(real_prob * 100)

        except Exception as e:
            print(f"[DeepfakeDetector] Error during inference: {e}")
            return self._mock_predict(image_bytes)

    def _mock_predict(self, image_bytes: bytes) -> int:
        """Deterministic mock based on image content hash."""
        h = int(hashlib.sha256(image_bytes[:256]).hexdigest(), 16)
        rng = random.Random(h)
        return rng.randint(20, 95)

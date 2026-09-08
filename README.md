# 🔍 TrustLens — AI Image Authenticity Detector

> Real-time deepfake & manipulation detection Chrome extension powered by multi-model AI fusion.

![TrustLens](https://img.shields.io/badge/TrustLens-v1.0.0-blue) ![Python](https://img.shields.io/badge/Python-3.10+-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-orange) ![Chrome](https://img.shields.io/badge/Chrome-MV3-yellow)

---

## 📁 Project Structure

```
trustlens/
├── extension/                  # Chrome Extension (MV3)
│   ├── manifest.json
│   ├── popup.html              # Extension popup UI
│   ├── js/
│   │   ├── content.js          # Injected page script (tooltips, panels)
│   │   ├── background.js       # Service worker
│   │   └── popup.js            # Popup logic
│   ├── css/
│   │   └── content.css         # Injected styles
│   └── icons/                  # Extension icons (add your own)
│
└── backend/                    # FastAPI Python Backend
    ├── main.py                 # API server + endpoints
    ├── requirements.txt
    ├── models/
    │   ├── deepfake_model.py   # EfficientNet-B4 wrapper
    │   ├── tampering_model.py  # ViT-B/16 wrapper
    │   ├── metadata_analyzer.py# EXIF analyzer
    │   └── fusion.py           # Score fusion engine
    └── utils/
        ├── heatmap.py          # Heatmap generator
        └── reverse_search.py   # Reverse image search
```

---

##  Quick Start

### 1. Backend Setup

```bash
cd trustlens/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

---

### 2. Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **"Load unpacked"**
4. Select the `trustlens/extension/` folder
5. The TrustLens icon will appear in your toolbar

> **Note:** You need placeholder icon PNGs in `extension/icons/`. Create 16×16, 48×48, and 128×128 pixel PNGs named `icon16.png`, `icon48.png`, `icon128.png`.

---

## 🔌 API Reference

### `GET /health`
Check server status and model availability.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": 3,
  "mock_mode": true
}
```

---

### `POST /analyze`
Analyze an image for authenticity.

**Request body:**
```json
{
  "image_url": "https://example.com/image.jpg"
}
```
Or with base64:
```json
{
  "image_b64": "<base64-encoded-image-bytes>"
}
```

**Response:**
```json
{
  "final_score": 34,
  "deepfake_score": 28,
  "tampering_score": 41,
  "metadata_flag": true,
  "heatmap_b64": "<base64-png>",
  "explanation": "Facial blending artifacts detected. EXIF metadata is inconsistent with claimed origin.",
  "source_info": "Similar image found in 2021 Reuters article about unrelated event.",
  "model_versions": {
    "deepfake": "EfficientNet-B4 (mock)",
    "tampering": "ViT-B/16 (mock)",
    "metadata": "EXIF-Analyzer v2",
    "fusion": "WeightedFusion v1"
  },
  "processing_time_ms": 142
}
```

**Score interpretation:**
| Score | Meaning |
|-------|---------|
| 75–100 | ✅ Likely Real |
| 45–74 | ⚠️ Suspicious |
| 0–44 | 🔴 Likely Fake |

---

### `POST /analyze/upload`
Upload an image file directly.

```bash
curl -X POST http://localhost:8000/analyze/upload \
  -F "file=@/path/to/image.jpg"
```

---

### `GET /demo`
Returns a canned demo response for testing the extension UI without a real image.

---

## 🤖 AI Models

### Deepfake Detection — EfficientNet-B4
- **Architecture:** EfficientNet-B4 (timm)
- **Dataset:** FaceForensics++ (when fine-tuned)
- **Task:** Binary classification — real vs. synthetic/manipulated face
- **Fallback:** Deterministic mock using image hash

**To use real weights:**
```bash
pip install timm torch torchvision
# Download pretrained weights from https://github.com/ondyari/FaceForensics
# Place at: backend/weights/efficientnet_b4_ff.pth
```

---

### Tampering Detection — ViT-B/16
- **Architecture:** Vision Transformer Base (HuggingFace)
- **Dataset:** CASIA v2 (when fine-tuned)
- **Task:** Pixel-level manipulation detection (splicing, clone-stamp, inpainting)
- **Bonus:** Attention maps used for heatmap generation

**To use real weights:**
```bash
pip install transformers
# Download fine-tuned CASIA weights or train on CASIA v2 dataset
# Place at: backend/weights/vit_tamper_casia.pth
```

---

### Metadata Analyzer — EXIF-Analyzer v2
- **Always active** (pure Python, no GPU needed)
- Checks: software used, camera make/model, datetime consistency, GPS presence
- Flags: Photoshop/GIMP edits, missing EXIF, timestamp mismatches

---

### Fusion Engine — WeightedFusion v1
Combines all model scores using configurable weights:

| Model | Default Weight |
|-------|---------------|
| Deepfake | 45% |
| Tampering | 40% |
| Metadata | 15% |

Rule-based overrides:
- If any model scores < 20 → cap final at 40
- Metadata flag applies −12 point penalty

---

##  Upgrading to Real Models

### Step 1: Install ML dependencies
```bash
pip install torch torchvision timm transformers
```

### Step 2: Download weights

**EfficientNet (FaceForensics++):**
```bash
mkdir -p backend/weights
# From the FaceForensics++ repo:
# https://github.com/ondyari/FaceForensics/tree/master/classification
```

**ViT (CASIA v2):**
Train or download from Hugging Face Hub:
```python
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained("your-org/vit-casia-tamper")
```

### Step 3: Uncomment model calls in `main.py`
```python
# In the /analyze endpoint:
if deepfake_detector:
    deepfake_score = await asyncio.to_thread(deepfake_detector.predict, image_bytes)
if tamper_detector:
    tampering_score = await asyncio.to_thread(tamper_detector.predict, image_bytes)
```

---

## 🌐 Real Reverse Search Integration

Replace `mock_reverse_search()` in `utils/reverse_search.py`:

**Option A: Google Cloud Vision**
```bash
pip install google-cloud-vision
export GOOGLE_CLOUD_API_KEY=your_key
```

**Option B: TinEye**
```bash
pip install tineye-api
```

**Option C: SerpAPI (Google Images)**
```bash
pip install google-search-results
```

---

## 🎨 Extension Features

| Feature | Description |
|---------|-------------|
| Auto-scan | Scans all images ≥ 80×80px on page load |
| Score badge | Color-coded % overlay on each image |
| Hover tooltip | Score ring, model bars, explanation |
| Click panel | Full analysis with heatmap, breakdown, actions |
| Toggle heatmap | Overlay showing suspicious regions |
| Reverse search | Opens Google Images reverse search |
| Copy report | Copies full report to clipboard |
| Page stats | Popup shows real/suspicious/fake counts |

---

## 💡 Accuracy Improvement Suggestions

1. **Fine-tune on recent datasets** — GAN landscapes evolve fast; retrain quarterly on DFDC, Celeb-DF v2
2. **Add CLIP consistency** — Cross-check image content vs. surrounding article text for context manipulation
3. **Ensemble more models** — Add Xception, ResNet-50, or CNNDetector for ensemble voting
4. **Video frame sampling** — Extract keyframes from `<video>` elements and analyze each
5. **Provenance chain** — Integrate C2PA/Content Credentials standard for camera-level signing
6. **User feedback loop** — Add thumbs up/down; fine-tune on human-corrected labels
7. **Browser caching** — Cache results in `chrome.storage.local` to avoid re-analyzing same URLs
8. **Domain reputation** — Weight scores by source domain credibility (Reuters vs. unknown blog)

---

## ⚙️ Configuration

Edit `backend/models/fusion.py` to tune weights:
```python
FusionConfig(
    deepfake_weight=0.50,   # Increase if deepfake model is better
    tampering_weight=0.35,
    metadata_weight=0.15,
    metadata_flag_penalty=15,
)
```

Edit `extension/js/content.js` to change API endpoint:
```javascript
const API_BASE = 'http://localhost:8000';  // Change for production
```

---

"""
TrustLens Backend API
─────────────────────
FastAPI server providing multi-model image authenticity analysis.
Uses pretrained models (or mocks) for deepfake detection, tampering
analysis, and metadata inspection.

Run: uvicorn main:app --reload --port 8000
"""

import io
import hashlib
import random
import time
import asyncio
import base64
from typing import Optional, Dict, Any
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

# ── Model imports (gracefully degrade to mocks) ─────────────────────────────
try:
    import torch
    import torchvision.transforms as T
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from .models.deepfake_model import DeepfakeDetector
    DEEPFAKE_MODEL_AVAILABLE = True
except ImportError:
    try:
        from models.deepfake_model import DeepfakeDetector
        DEEPFAKE_MODEL_AVAILABLE = True
    except ImportError:
        DEEPFAKE_MODEL_AVAILABLE = False

try:
    from .models.tampering_model import TamperingDetector
    TAMPER_MODEL_AVAILABLE = True
except ImportError:
    try:
        from models.tampering_model import TamperingDetector
        TAMPER_MODEL_AVAILABLE = True
    except ImportError:
        TAMPER_MODEL_AVAILABLE = False

try:
    from .models.metadata_analyzer import MetadataAnalyzer
    from .models.fusion import FusionEngine
    from .utils.heatmap import generate_heatmap_base64
    from .utils.reverse_search import mock_reverse_search
except ImportError:
    from models.metadata_analyzer import MetadataAnalyzer
    from models.fusion import FusionEngine
    from utils.heatmap import generate_heatmap_base64
    from utils.reverse_search import mock_reverse_search

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrustLens API",
    description="Multi-model AI image authenticity analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models (initialized at startup) ──────────────────────────────────────────
metadata_analyzer = MetadataAnalyzer()
fusion_engine = FusionEngine()
deepfake_detector = None
tamper_detector = None

@app.on_event("startup")
async def startup():
    global deepfake_detector, tamper_detector
    print("[TrustLens] Loading models...")
    if DEEPFAKE_MODEL_AVAILABLE:
        deepfake_detector = DeepfakeDetector()
        print("[TrustLens] ✓ Deepfake model loaded")
    else:
        print("[TrustLens] ⚠ Deepfake model unavailable — using mock")

    if TAMPER_MODEL_AVAILABLE:
        tamper_detector = TamperingDetector()
        print("[TrustLens] ✓ Tampering model loaded")
    else:
        print("[TrustLens] ⚠ Tampering model unavailable — using mock")
    print("[TrustLens] Ready.")


# ── Request / Response schemas ────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    image_url: Optional[str] = None
    image_b64: Optional[str] = None  # base64-encoded image bytes


class ModelScores(BaseModel):
    deepfake_score: int          # 0=fake, 100=real
    tampering_score: int
    metadata_flag: bool
    metadata_details: Dict[str, Any]


class AnalysisResponse(BaseModel):
    final_score: int             # 0=fake, 100=real
    deepfake_score: int
    tampering_score: int
    metadata_flag: bool
    heatmap_b64: Optional[str]
    explanation: str
    source_info: str
    model_versions: Dict[str, str]
    processing_time_ms: int


# ── Helpers ───────────────────────────────────────────────────────────────────
async def fetch_image_bytes(url: str) -> bytes:
    """Download image from URL with timeout."""
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        resp = await client.get(url, headers={"User-Agent": "TrustLens/1.0"})
        resp.raise_for_status()
        return resp.content


def deterministic_mock_scores(seed_bytes: bytes) -> Dict[str, Any]:
    """
    Deterministic mock scores based on image hash.
    Replace each section with real model calls once models are integrated.
    """
    h = int(hashlib.md5(seed_bytes[:512] if len(seed_bytes) > 512 else seed_bytes).hexdigest(), 16)
    rng = random.Random(h)

    deepfake = rng.randint(20, 95)
    tampering = rng.randint(15, 95)

    # Introduce correlation: if one is low, the other tends to be low
    if deepfake < 50:
        tampering = max(10, tampering - rng.randint(5, 20))

    return {
        "deepfake_score": deepfake,
        "tampering_score": tampering,
    }


EXPLANATIONS = {
    "fake": [
        "Facial blending artifacts detected in the upper region. GAN fingerprint patterns identified.",
        "Clone-stamp manipulation found near background edges. Frequency domain shows splicing signatures.",
        "JPEG block inconsistencies suggest heavy post-processing. Face geometry deviates from natural proportions.",
    ],
    "warn": [
        "Mild compression inconsistencies detected. Could indicate cropping or re-saving from social media.",
        "Some noise pattern irregularities found. Metadata is partially missing.",
        "Lighting direction inconsistencies between foreground and background regions.",
    ],
    "real": [
        "No manipulation signals detected. Image appears authentic.",
        "EXIF data consistent with original capture device. No GAN fingerprints found.",
        "Natural noise patterns and compression artifacts consistent with unmodified photograph.",
    ]
}


def build_explanation(final_score: int, deepfake_score: int, tampering_score: int, metadata_flag: bool) -> str:
    if final_score < 45:
        tier = "fake"
    elif final_score < 75:
        tier = "warn"
    else:
        tier = "real"

    h = (deepfake_score * 7 + tampering_score * 3) % len(EXPLANATIONS[tier])
    explanation = EXPLANATIONS[tier][h]

    if metadata_flag:
        explanation += " EXIF metadata is missing or inconsistent with claimed origin."
    return explanation


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": sum([
            DEEPFAKE_MODEL_AVAILABLE,
            TAMPER_MODEL_AVAILABLE,
            True  # metadata always available
        ]),
        "mock_mode": not (DEEPFAKE_MODEL_AVAILABLE and TAMPER_MODEL_AVAILABLE),
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalyzeRequest):
    start = time.perf_counter()

    # ── 1. Load image ──────────────────────────────────────────────────────
    image_bytes: Optional[bytes] = None

    if req.image_url:
        try:
            image_bytes = await fetch_image_bytes(req.image_url)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not fetch image: {e}")

    elif req.image_b64:
        try:
            image_bytes = base64.b64decode(req.image_b64)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid base64 image data")

    else:
        raise HTTPException(status_code=400, detail="Provide image_url or image_b64")

    # ── 2. Run models (parallel) ───────────────────────────────────────────
    scores = deterministic_mock_scores(image_bytes)
    deepfake_score = scores["deepfake_score"]
    tampering_score = scores["tampering_score"]

    if deepfake_detector:
        deepfake_score = await asyncio.to_thread(deepfake_detector.predict, image_bytes)
    if tamper_detector:
        tampering_score = await asyncio.to_thread(tamper_detector.predict, image_bytes)

    # ── 3. Metadata analysis ───────────────────────────────────────────────
    meta = metadata_analyzer.analyze(image_bytes)

    # ── 4. Fusion ─────────────────────────────────────────────────────────
    final_score = fusion_engine.fuse(
        deepfake_score=deepfake_score,
        tampering_score=tampering_score,
        metadata_flag=meta["flag"],
    )

    # ── 5. Heatmap ────────────────────────────────────────────────────────
    heatmap_b64 = generate_heatmap_base64(image_bytes, final_score)

    # ── 6. Reverse search mock ─────────────────────────────────────────────
    source_info = mock_reverse_search(req.image_url or "unknown")

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return AnalysisResponse(
        final_score=final_score,
        deepfake_score=deepfake_score,
        tampering_score=tampering_score,
        metadata_flag=meta["flag"],
        heatmap_b64=heatmap_b64,
        explanation=build_explanation(final_score, deepfake_score, tampering_score, meta["flag"]),
        source_info=source_info,
        model_versions={
            "deepfake": "EfficientNet-B4 (mock)" if not DEEPFAKE_MODEL_AVAILABLE else "EfficientNet-B4",
            "tampering": "ViT-B/16 (mock)" if not TAMPER_MODEL_AVAILABLE else "ViT-B/16",
            "metadata": "EXIF-Analyzer v2",
            "fusion": "WeightedFusion v1",
        },
        processing_time_ms=elapsed_ms,
    )


@app.post("/analyze/upload")
async def analyze_upload(file: UploadFile = File(...)):
    """Analyze an uploaded image file directly."""
    image_bytes = await file.read()
    req = AnalyzeRequest(image_b64=base64.b64encode(image_bytes).decode())
    return await analyze(req)


@app.get("/demo")
async def demo():
    """Return a demo analysis response for testing."""
    return {
        "final_score": 34,
        "deepfake_score": 28,
        "tampering_score": 41,
        "metadata_flag": True,
        "heatmap_b64": None,
        "explanation": "Facial blending artifacts detected. EXIF metadata is inconsistent with claimed origin.",
        "source_info": "Similar image found in 2021 Reuters article about unrelated event.",
        "model_versions": {
            "deepfake": "EfficientNet-B4 (mock)",
            "tampering": "ViT-B/16 (mock)",
            "metadata": "EXIF-Analyzer v2",
            "fusion": "WeightedFusion v1",
        },
        "processing_time_ms": 142,
    }

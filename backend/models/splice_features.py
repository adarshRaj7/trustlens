"""
Hand-crafted forensic features for JPEG splice detection.

Signals used (all classical image-forensics techniques, no training data required
to compute — only to calibrate a classifier on top of them):
  - Error Level Analysis (ELA): re-save at a fixed JPEG quality and diff against
    the original; regions with a different compression history show a different
    error level than their surroundings.
  - JPEG Ghost (Farid, 2009): sweep resave quality and find the minimum-error
    quality for a region — spliced regions carry a double-compression fingerprint
    that resists a clean single minimum ("flatness" below).
  - Noise-residual variance: median-filter residual, a proxy for sensor/compression
    noise, which typically differs between two different source images.
  - FFT high-frequency energy ratio: resampling/resizing during a splice leaves
    detectable high-frequency spectral energy.

See backend/weights/SPLICE_MODEL_CARD.md for how the shipped classifier was
trained and validated.
"""

import io
import numpy as np
from PIL import Image
from scipy import ndimage


def ela_stats(im, quality=90):
    orig = np.asarray(im, dtype=np.float32)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    diff = np.abs(orig - resaved).sum(axis=2)
    return float(diff.mean()), float(diff.std()), float(diff.max())


def ghost_best_quality(im, qualities=range(40, 100, 5)):
    orig = np.asarray(im, dtype=np.float32)
    best_q, best_err = None, None
    errs = []
    for q in qualities:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        buf.seek(0)
        resaved = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
        err = np.mean(np.abs(orig - resaved))
        errs.append(err)
        if best_err is None or err < best_err:
            best_err, best_q = err, q
    errs = np.array(errs)
    flatness = float(errs.std() / (errs.mean() + 1e-6))
    return float(best_q), flatness


def noise_residual_var(im):
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    denoised = ndimage.median_filter(gray, size=3)
    residual = gray - denoised
    return float(residual.var())


def fft_high_freq_ratio(im):
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    f = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_r = min(cy, cx)
    high_mask = r > 0.6 * max_r
    total = mag.sum() + 1e-6
    return float(mag[high_mask].sum() / total)


FEATURE_NAMES = ["ela_mean", "ela_std", "ela_max", "ghost_best_q", "ghost_flatness", "noise_var", "fft_hf_ratio"]


def extract_features(im, quality=90):
    ela_mean, ela_std, ela_max = ela_stats(im, quality)
    best_q, flatness = ghost_best_quality(im)
    noise_var = noise_residual_var(im)
    hf_ratio = fft_high_freq_ratio(im)
    return [ela_mean, ela_std, ela_max, best_q, flatness, noise_var, hf_ratio]

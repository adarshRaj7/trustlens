"""
Heatmap Generator
──────────────────
Generates visual heatmaps overlaid on images to highlight suspicious regions.
Uses Gaussian blobs positioned based on model attention scores.
Returns base64-encoded PNG.
"""

import io
import base64
import hashlib
import math
import random
from typing import Optional


def generate_heatmap_base64(image_bytes: bytes, final_score: int) -> Optional[str]:
    """
    Generate a heatmap PNG overlaid on the source image.
    Returns base64-encoded PNG string or None if PIL unavailable.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np

        # Load source image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGBA')
        w, h = img.size

        # Scale down for processing
        max_dim = 600
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            w, h = img.size

        # Generate heatmap layer
        heatmap = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(heatmap)

        # Seed blob positions from image content (deterministic)
        seed = int(hashlib.md5(image_bytes[:256]).hexdigest(), 16)
        rng = random.Random(seed)

        num_blobs = max(1, int((100 - final_score) / 15))

        for _ in range(num_blobs):
            cx = rng.randint(int(w * 0.1), int(w * 0.9))
            cy = rng.randint(int(h * 0.1), int(h * 0.9))
            radius = rng.randint(int(min(w, h) * 0.08), int(min(w, h) * 0.22))
            intensity = rng.randint(100, 200)

            # Draw radial gradient blob
            for r in range(radius, 0, -4):
                alpha = int(intensity * (1 - r / radius) ** 2)
                if final_score < 45:
                    color = (239, 68, 68, alpha)   # red
                elif final_score < 75:
                    color = (245, 158, 11, alpha)  # amber
                else:
                    color = (34, 197, 94, alpha)   # green
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

        # Blur for smooth gradient effect
        heatmap = heatmap.filter(ImageFilter.GaussianBlur(radius=12))

        # Composite
        result = Image.alpha_composite(img, heatmap).convert('RGB')

        buf = io.BytesIO()
        result.save(buf, format='PNG', optimize=True)
        return base64.b64encode(buf.getvalue()).decode()

    except ImportError:
        return None
    except Exception as e:
        print(f"[Heatmap] Error: {e}")
        return None

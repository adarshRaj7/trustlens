"""
Metadata Analyzer
──────────────────
Extracts and validates EXIF metadata from images.
Flags inconsistencies that suggest manipulation or redistribution.
"""

import io
from typing import Dict, Any


SUSPICIOUS_SOFTWARE = {
    "adobe photoshop", "gimp", "paint.net", "affinity photo",
    "pixelmator", "canva", "snapseed", "facetune", "meitu",
    "lightroom", "capture one", "darktable",
}

TRUSTED_CAMERAS = {
    "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
    "leica", "hasselblad", "pentax", "apple", "samsung", "google",
}


class MetadataAnalyzer:
    """EXIF metadata extraction and anomaly detection."""

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze image metadata.
        Returns dict with 'flag' (bool) and 'details' (dict).
        """
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            img = Image.open(io.BytesIO(image_bytes))
            exif_data = img._getexif()

            if exif_data is None:
                return {
                    "flag": True,
                    "reason": "No EXIF data found — stripped or generated image",
                    "details": {"exif_present": False},
                }

            tags = {TAGS.get(k, k): v for k, v in exif_data.items()}
            return self._evaluate_tags(tags)

        except ImportError:
            return self._mock_analyze(image_bytes)
        except Exception as e:
            return {
                "flag": False,
                "reason": f"Could not parse metadata: {e}",
                "details": {},
            }

    def _evaluate_tags(self, tags: dict) -> Dict[str, Any]:
        flags = []
        details = {}

        # Check software
        software = str(tags.get("Software", "")).lower()
        details["software"] = software or None
        if any(s in software for s in SUSPICIOUS_SOFTWARE):
            flags.append(f"Edited with: {software}")

        # Check make/model
        make = str(tags.get("Make", "")).lower()
        model = str(tags.get("Model", ""))
        details["camera"] = f"{make} {model}".strip() or None
        if make and not any(c in make for c in TRUSTED_CAMERAS):
            flags.append(f"Unknown camera make: {make}")

        # DateTime consistency
        orig_time = tags.get("DateTimeOriginal")
        mod_time = tags.get("DateTime")
        details["datetime_original"] = str(orig_time) if orig_time else None
        details["datetime_modified"] = str(mod_time) if mod_time else None
        if orig_time and mod_time and orig_time != mod_time:
            flags.append("Modification timestamp differs from capture time")

        # GPS present (can indicate original)
        gps = tags.get("GPSInfo")
        details["has_gps"] = gps is not None

        # Thumbnail vs main image
        thumbnail = tags.get("ThumbnailOffset")
        details["has_thumbnail"] = thumbnail is not None

        return {
            "flag": len(flags) > 0,
            "reason": "; ".join(flags) if flags else "Metadata appears consistent",
            "details": details,
            "flag_count": len(flags),
        }

    def _mock_analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        """Mock analysis when PIL not available."""
        import hashlib, random
        h = int(hashlib.md5(image_bytes[:64]).hexdigest(), 16)
        flag = random.Random(h + 99).random() > 0.6
        return {
            "flag": flag,
            "reason": "Missing EXIF data (mock)" if flag else "Metadata OK (mock)",
            "details": {"mock": True},
        }

"""
Fusion Engine
──────────────
Combines scores from multiple models into a single authenticity score.
Uses weighted average with configurable weights and rule-based overrides.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FusionConfig:
    # Weights must sum to 1.0
    deepfake_weight: float = 0.45
    tampering_weight: float = 0.40
    metadata_weight: float = 0.15

    # Rule-based overrides
    # If any single model is THIS confident of fake, cap the score
    single_model_fake_threshold: int = 20  # score below this → cap final at 40
    single_model_cap: int = 40

    # Metadata flag penalty
    metadata_flag_penalty: int = 12


class FusionEngine:
    """
    Weighted fusion of deepfake, tampering, and metadata signals.
    """

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fuse(
        self,
        deepfake_score: int,
        tampering_score: int,
        metadata_flag: bool,
    ) -> int:
        """
        Returns final authenticity score 0–100.
        Higher = more likely authentic.
        """
        cfg = self.config

        # Convert metadata flag to a numeric signal
        metadata_score = 30 if metadata_flag else 90

        # Weighted average
        weighted = (
            deepfake_score * cfg.deepfake_weight
            + tampering_score * cfg.tampering_weight
            + metadata_score * cfg.metadata_weight
        )

        final = int(weighted)

        # Apply penalty for metadata flag
        if metadata_flag:
            final = max(0, final - cfg.metadata_flag_penalty)

        # Rule-based override: if any model is very confident of fake
        if deepfake_score < cfg.single_model_fake_threshold:
            final = min(final, cfg.single_model_cap)
        if tampering_score < cfg.single_model_fake_threshold:
            final = min(final, cfg.single_model_cap)

        return max(0, min(100, final))

    def explain_fusion(
        self,
        deepfake_score: int,
        tampering_score: int,
        metadata_flag: bool,
        final_score: int,
    ) -> str:
        """Generate a human-readable fusion explanation."""
        parts = []
        cfg = self.config

        parts.append(
            f"Deepfake model ({cfg.deepfake_weight*100:.0f}% weight): {deepfake_score}/100"
        )
        parts.append(
            f"Tampering model ({cfg.tampering_weight*100:.0f}% weight): {tampering_score}/100"
        )
        if metadata_flag:
            parts.append(f"Metadata: flagged (−{cfg.metadata_flag_penalty} penalty)")
        else:
            parts.append("Metadata: clean")

        parts.append(f"→ Final score: {final_score}/100")
        return " | ".join(parts)

"""
TinyAya model configurations.

Training-stage ordering:
  Base → Global → Regional (Earth / Fire / Water)

Earth / Fire / Water are hypothetical regional variant names; update
HF_IDS once the actual repo slugs are confirmed with Cohere.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    name: str                      # Short identifier used in filenames
    hf_id: str                     # HuggingFace model ID
    stage: str                     # "base" | "global" | "regional"
    region: Optional[str] = None   # e.g. "africa", "asia", "latam"
    # Layers to extract residual-stream activations from (0-indexed)
    probe_layers: list[int] = field(default_factory=lambda: [18, 24, 30])
    n_layers: int = 32             # Update once model cards are available


MODELS: dict[str, ModelConfig] = {
    "base": ModelConfig(
        name="TinyAya-Base",
        hf_id="CohereForAI/aya-expanse-8b",   # placeholder — update on access
        stage="base",
        probe_layers=[18, 24, 30],
    ),
    "global": ModelConfig(
        name="TinyAya-Global",
        hf_id="CohereForAI/aya-expanse-8b",   # placeholder
        stage="global",
        probe_layers=[18, 24, 30],
    ),
    "earth": ModelConfig(
        name="TinyAya-Earth",
        hf_id="CohereForAI/aya-expanse-8b",   # placeholder
        stage="regional",
        region="africa",
        probe_layers=[18, 24, 30],
    ),
    "fire": ModelConfig(
        name="TinyAya-Fire",
        hf_id="CohereForAI/aya-expanse-8b",   # placeholder
        stage="regional",
        region="asia",
        probe_layers=[18, 24, 30],
    ),
    "water": ModelConfig(
        name="TinyAya-Water",
        hf_id="CohereForAI/aya-expanse-8b",   # placeholder
        stage="regional",
        region="latam",
        probe_layers=[18, 24, 30],
    ),
}

# Ordered by training stage for plots
MODEL_ORDER = ["base", "global", "earth", "fire", "water"]

# Layer used for SAE training (mid-late layer; update after probe sweep)
SAE_LAYER = 24

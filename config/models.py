"""
TinyAya model configurations plus comparison baselines.

Training-stage ordering:
  Base → Global

Comparison baselines (from Phase Work scope / MultiJail table):
  Gemma3-4B, Ministral-3-3B, Qwen3-4B
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
        hf_id="CohereLabs/tiny-aya-base",
        stage="base",
        probe_layers=[4, 9, 18, 24, 30, 35],  # 35 = last layer (36 layers, 0-indexed)
        n_layers=36,
    ),
    "global": ModelConfig(
        name="TinyAya-Global",
        hf_id="CohereLabs/tiny-aya-global",
        stage="global",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_sft": ModelConfig(
        name="TinyAya-Base-SFT",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned/base_sft",
        stage="finetuned",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_dpo": ModelConfig(
        name="TinyAya-Base-DPO",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned/base_dpo",
        stage="finetuned",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_rlhf": ModelConfig(
        name="TinyAya-Base-RLHF",
        hf_id="/workspace/cohere-safety-representations/models/finetuned/base_rlhf",  # local adapter path
        stage="finetuned",
        probe_layers=[18, 24, 30],
    ),
    # ── Comparison baselines (Phase Work scope) ──────────────────────────
    "gemma": ModelConfig(
        name="Gemma3-4B",
        hf_id="google/gemma-3-4b-it",
        stage="baseline",
        probe_layers=[4, 9, 18, 24, 30],
        n_layers=34,
    ),
    "ministral": ModelConfig(
        name="Ministral-3-3B",
        hf_id="mistralai/Ministral-3B-Instruct-2410",
        stage="baseline",
        probe_layers=[4, 9, 18, 24, 30, 36],
        n_layers=36,
    ),
    "qwen": ModelConfig(
        name="Qwen3-4B",
        hf_id="Qwen/Qwen3-4B",
        stage="baseline",
        probe_layers=[4, 9, 18, 24, 30, 36],
        n_layers=36,
    ),
}

# Ordered by training stage for plots
MODEL_ORDER = ["base", "global", "base_sft", "base_dpo", "base_rlhf", "gemma", "ministral", "qwen"]

# Layer used for SAE training (mid-late layer; update after probe sweep)
SAE_LAYER = 24

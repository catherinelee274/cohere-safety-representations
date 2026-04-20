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
        name="TinyAya-Base-SFT-500",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned/base_sft",
        stage="finetuned",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_sft_20k": ModelConfig(
        name="TinyAya-Base-SFT-20k",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned_20k/base_sft",
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
    # ── SFT data-size sweep (finetuned_sweep/) ───────────────────────────
    "base_sft_n500": ModelConfig(
        name="TinyAya-Base-SFT-n500",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned_sweep/base_sft_n500",
        stage="finetuned",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_sft_n2000": ModelConfig(
        name="TinyAya-Base-SFT-n2000",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned_sweep/base_sft_n2000",
        stage="finetuned",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_sft_n5000": ModelConfig(
        name="TinyAya-Base-SFT-n5000",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned_sweep/base_sft_n5000",
        stage="finetuned",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_sft_n10000": ModelConfig(
        name="TinyAya-Base-SFT-n10000",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned_sweep/base_sft_n10000",
        stage="finetuned",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "base_sft_n20000": ModelConfig(
        name="TinyAya-Base-SFT-n20000",
        hf_id="/home/catherine/Documents/GitHub/cohere-safety-representations/models/finetuned_sweep/base_sft_n20000",
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
    # ── Regional variants ────────────────────────────────────────────────────
    "earth": ModelConfig(
        name="TinyAya-Earth",
        hf_id="CohereLabs/tiny-aya-earth",
        stage="regional",
        region="africa_westasia",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "fire": ModelConfig(
        name="TinyAya-Fire",
        hf_id="CohereLabs/tiny-aya-fire",
        stage="regional",
        region="south_asia",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "water": ModelConfig(
        name="TinyAya-Water",
        hf_id="CohereLabs/tiny-aya-water",
        stage="regional",
        region="asiapacific_europe",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
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
        hf_id="mistralai/Ministral-3-3B-Instruct-2512",
        stage="baseline",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
    "qwen": ModelConfig(
        name="Qwen3-4B",
        hf_id="Qwen/Qwen3-4B",
        stage="baseline",
        probe_layers=[4, 9, 18, 24, 30, 35],
        n_layers=36,
    ),
}

# Ordered by training stage for plots
MODEL_ORDER = ["base", "global", "earth", "fire", "water",
               "base_sft", "base_sft_20k", "base_dpo",
               "base_sft_n500", "base_sft_n2000", "base_sft_n5000", "base_sft_n10000", "base_sft_n20000",
               "gemma", "ministral", "qwen"]

# Layer used for SAE training (mid-late layer; update after probe sweep)
SAE_LAYER = 24


def get_checkpoint_configs(base_key: str) -> list[tuple[str, "ModelConfig"]]:
    """
    Discover all checkpoint-* subdirectories under the base model's hf_id path
    and return a list of (checkpoint_key, ModelConfig) sorted by step number.

    The checkpoint_key format is ``{base_key}_ckpt{step}``, e.g.
    ``base_sft_n5000_ckpt500``.  Activations for each checkpoint are expected
    at ``data/processed/activations/{checkpoint_key}/``.

    Returns an empty list if the base model's hf_id directory does not exist or
    contains no checkpoint-* subdirectories.
    """
    from pathlib import Path
    import re

    cfg = MODELS.get(base_key)
    if cfg is None:
        raise KeyError(f"Unknown model key: {base_key!r}")

    model_dir = Path(cfg.hf_id)
    if not model_dir.is_dir():
        return []

    results = []
    for ckpt_dir in sorted(model_dir.glob("checkpoint-*")):
        m = re.fullmatch(r"checkpoint-(\d+)", ckpt_dir.name)
        if m is None:
            continue
        step = int(m.group(1))
        ckpt_key = f"{base_key}_ckpt{step}"
        ckpt_cfg = ModelConfig(
            name=f"{cfg.name}-ckpt{step}",
            hf_id=str(ckpt_dir),
            stage=cfg.stage,
            region=cfg.region,
            probe_layers=cfg.probe_layers,
            n_layers=cfg.n_layers,
        )
        results.append((step, ckpt_key, ckpt_cfg))

    results.sort(key=lambda t: t[0])
    return [(ckpt_key, ckpt_cfg) for _, ckpt_key, ckpt_cfg in results]

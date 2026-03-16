"""
Step 1 — Verify HuggingFace access to TinyAya variants.

Usage:
    python scripts/01_check_model_access.py [--token YOUR_HF_TOKEN]

Prints a summary table of which models are accessible and what
their actual layer counts are (so you can update config/models.py).
"""

import argparse
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.models import MODELS


def check_model(hf_id: str, token: str | None) -> dict:
    """Return access info for a single HuggingFace model."""
    try:
        from huggingface_hub import model_info
        info = model_info(hf_id, token=token)
        return {"accessible": True, "id": info.id, "private": info.private}
    except Exception as e:
        return {"accessible": False, "error": str(e)}


def load_config(hf_id: str, token: str | None) -> dict:
    """Try to load model config to read n_layers."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(hf_id, token=token)
        n_layers = getattr(cfg, "num_hidden_layers", None)
        hidden = getattr(cfg, "hidden_size", None)
        return {"n_layers": n_layers, "hidden_size": hidden}
    except Exception as e:
        return {"n_layers": None, "hidden_size": None, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--use-hf-mirror", action="store_true",
                        help="Use Hugging Face mirror as source for models")
    args = parser.parse_args()

    # Set HF endpoint to mirror if specified
    if args.use_hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using Hugging Face mirror: https://hf-mirror.com")

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"\n{'Model':<20} {'HF ID':<45} {'Access':<10} {'Layers':<8} {'Hidden'}")
    print("-" * 100)

    all_ok = True
    for key, cfg in MODELS.items():
        info   = check_model(cfg.hf_id, token)
        linfo  = load_config(cfg.hf_id, token) if info["accessible"] else {}
        status = "OK" if info["accessible"] else f"FAIL: {info.get('error','?')[:30]}"
        n      = linfo.get("n_layers", "?")
        h      = linfo.get("hidden_size", "?")
        if not info["accessible"]:
            all_ok = False
        print(f"{cfg.name:<20} {cfg.hf_id:<45} {status:<10} {str(n):<8} {h}")

    print()
    if all_ok:
        print("All models accessible. Update config/models.py with layer counts above.")
    else:
        print("Some models inaccessible. Request access at https://huggingface.co/<model>")
        print("Then re-run this script before proceeding.")
    print()


if __name__ == "__main__":
    main()

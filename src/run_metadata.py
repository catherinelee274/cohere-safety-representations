"""
Shared utility for writing run_info.json alongside eval outputs.

Every script that produces results calls save_run_info() so that any output
file can be traced back to: which model (and exact local path / HF id), when
it was run, which git commit, and what the key eval parameters were.

Usage:
    from src.run_metadata import save_run_info, model_meta

    save_run_info(out_dir, {
        **model_meta("global"),
        "script": "05_extract_activations",
        "lang": "en",
        "n_harmful": 430,
        "n_harmless": 430,
        "layers": [4, 9, 18, 24, 30, 35],
    })
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config.models import MODELS, ModelConfig


def _git_commit() -> str | None:
    """Return current HEAD short hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def model_meta(model_key: str) -> dict:
    """Return standard traceability fields for a model key."""
    cfg: ModelConfig = MODELS[model_key]
    return {
        "model_key":    model_key,
        "model_name":   cfg.name,
        "model_hf_id":  cfg.hf_id,
        "model_stage":  cfg.stage,
    }


def save_run_info(out_dir: Path, info: dict) -> Path:
    """
    Write run_info.json to out_dir, merging in run_at and git_commit.

    If run_info.json already exists the new entry is appended to a list,
    so repeated runs on the same directory are fully traceable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "run_info.json"

    entry = {
        "run_at":     datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        **info,
    }

    # Append to existing history so no run is lost
    history: list[dict] = []
    if info_path.exists():
        try:
            existing = json.loads(info_path.read_text())
            history = existing if isinstance(existing, list) else [existing]
        except Exception:
            pass
    history.append(entry)

    info_path.write_text(json.dumps(history, indent=2))
    return info_path

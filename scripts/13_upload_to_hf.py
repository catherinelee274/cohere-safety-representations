"""
Step 13 — Upload fine-tuned models / adapters to HuggingFace Hub.

Reads model paths from config/models.py (any entry whose hf_id is a local
path, i.e. starts with "/" or ".") and pushes them to the Hub under a
specified org/user namespace.

A README model card is auto-generated from training_info.json written by
scripts/10_finetune.py. If no training_info.json is found, a minimal card
is written instead.

Usage:
    # Upload a single model key
    python scripts/13_upload_to_hf.py --model base_sft --repo-prefix catherine-safety

    # Upload all local models (auto-detect)
    python scripts/13_upload_to_hf.py --all --repo-prefix catherine-safety

    # Dry run to see what would be uploaded
    python scripts/13_upload_to_hf.py --all --repo-prefix catherine-safety --dry-run

    # Upload with explicit repo name (overrides auto-naming)
    python scripts/13_upload_to_hf.py --model base_sft --repo-id org/my-repo-name

    # Upload as private repo
    python scripts/13_upload_to_hf.py --model base_dpo --repo-prefix catherine-safety --private

    # Upload only LoRA adapter files (not full model weights)
    python scripts/13_upload_to_hf.py --model base_sft --repo-prefix catherine-safety --adapter-only

Requirements:
    pip install huggingface_hub transformers peft
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.models import MODELS, ModelConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── model card ────────────────────────────────────────────────────────────────

def _load_training_info(local_path: Path) -> dict | None:
    info_path = local_path / "training_info.json"
    if not info_path.exists():
        return None
    with open(info_path) as f:
        return json.load(f)


def build_model_card(model_cfg: ModelConfig, local_path: Path, repo_id: str) -> str:
    """Return a README.md string for this model."""
    info = _load_training_info(local_path)
    upload_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    lines = [
        "---",
        f"base_model: {model_cfg.hf_id if not _is_local(model_cfg.hf_id) else 'unknown'}",
        "library_name: peft",
        "tags:",
        "  - safety",
        "  - multilingual",
        "  - lora",
        "  - cohere-safety-representations",
        "---",
        "",
        f"# {model_cfg.name}",
        "",
        f"> Uploaded: {upload_date}  ",
        f"> Project: [cohere-safety-representations](https://github.com/CathyLee/cohere-safety-representations)",
        "",
    ]

    if info:
        trained_at = info.get("trained_at", "unknown")
        method = info.get("method", "unknown").upper()
        base = info.get("base_model_hf_id", "unknown")
        n = info.get("num_examples", "unknown")
        data_src = info.get("data_source", "unknown")
        lang = info.get("lang")
        hp = info.get("hyperparams", {})

        lines += [
            "## Training details",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Trained at (UTC) | `{trained_at}` |",
            f"| Method | {method} |",
            f"| Base model | `{base}` |",
            f"| Training examples | {n} |",
            f"| Data source | {data_src}{' (' + lang + ')' if lang else ''} |",
            "",
            "### Hyperparameters",
            "",
            "| Hyperparameter | Value |",
            "|----------------|-------|",
        ]
        for k, v in hp.items():
            lines.append(f"| `{k}` | {v} |")

        lines += [
            "",
            "### LoRA targets",
            "",
            "```",
            "q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj",
            "```",
            "",
        ]
    else:
        lines += [
            "## Training details",
            "",
            "_No `training_info.json` found in the adapter directory._  ",
            "_Re-run `scripts/10_finetune.py` to generate it automatically._",
            "",
        ]

    lines += [
        "## Usage",
        "",
        "```python",
        "from transformers import AutoTokenizer, AutoModelForCausalLM",
        "from peft import PeftModel",
        "",
        f'base = AutoModelForCausalLM.from_pretrained("{info.get("base_model_hf_id", "<base-model-id>") if info else "<base-model-id>"}")',
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")',
        f'model = PeftModel.from_pretrained(base, "{repo_id}")',
        "```",
        "",
        "## Probe layers",
        "",
        f"Layers used for activation probing: `{model_cfg.probe_layers}`",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{cohere-safety-representations,",
        "  title  = {Safety Representations in TinyAya Multilingual Models},",
        "  author = {Catherine Lee},",
        "  year   = {2025},",
        "}",
        "```",
    ]

    return "\n".join(lines) + "\n"


def _is_local(hf_id: str) -> bool:
    return hf_id.startswith("/") or hf_id.startswith(".")


# ── upload helpers ────────────────────────────────────────────────────────────

def is_local_path(hf_id: str) -> bool:
    return hf_id.startswith("/") or hf_id.startswith(".")


def local_models() -> dict[str, ModelConfig]:
    return {k: v for k, v in MODELS.items() if is_local_path(v.hf_id)}


def repo_id_for(model_cfg: ModelConfig, prefix: str) -> str:
    """Build '<prefix>/<slug>' from the model name."""
    slug = model_cfg.name.lower().replace(" ", "-")
    return f"{prefix}/{slug}"


def upload_model(
    model_key: str,
    model_cfg: ModelConfig,
    repo_id: str,
    private: bool,
    adapter_only: bool,
    dry_run: bool,
) -> None:
    local_path = Path(model_cfg.hf_id)
    if not local_path.exists():
        log.error("Local path does not exist: %s  (model key: %s)", local_path, model_key)
        return

    log.info("Model  : %s  (%s)", model_key, model_cfg.name)
    log.info("Source : %s", local_path)
    log.info("Target : %s  (private=%s, adapter_only=%s)", repo_id, private, adapter_only)

    # Always write model card to disk first (even in dry-run, for preview)
    readme_path = local_path / "README.md"
    card_text = build_model_card(model_cfg, local_path, repo_id)
    readme_path.write_text(card_text)
    log.info("Model card written → %s", readme_path)

    if dry_run:
        log.info("[DRY RUN] Model card preview:\n%s", card_text[:800])
        log.info("[DRY RUN] Skipping actual upload.")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()

    # Create repo if it doesn't exist yet
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        log.info("Repo already exists: %s", repo_id)
    except RepositoryNotFoundError:
        log.info("Creating repo: %s", repo_id)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private)

    # Upload README first so it appears immediately
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    log.info("README uploaded.")

    if adapter_only:
        _upload_adapter(api, local_path, repo_id)
    else:
        _upload_full(api, local_path, repo_id, model_cfg)

    log.info("Done: https://huggingface.co/%s", repo_id)


def _upload_adapter(api, local_path: Path, repo_id: str) -> None:
    """Push LoRA adapter files + README."""
    adapter_files = (
        list(local_path.glob("adapter_config.json"))
        + list(local_path.glob("adapter_model.*"))
        + list(local_path.glob("tokenizer*"))
        + list(local_path.glob("special_tokens_map.json"))
        + list(local_path.glob("training_info.json"))
    )

    if not adapter_files:
        log.warning("No adapter files found in %s; falling back to full folder upload.", local_path)
        _upload_full(api, local_path, repo_id, model_cfg=None)
        return

    for fpath in sorted(adapter_files):
        log.info("  Uploading %s", fpath.name)
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fpath.name,
            repo_id=repo_id,
            repo_type="model",
        )


def _upload_full(api, local_path: Path, repo_id: str, model_cfg) -> None:
    """Push the entire folder (large models may take a while)."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        log.info("Loading model from %s …", local_path)
        tokenizer = AutoTokenizer.from_pretrained(str(local_path))

        adapter_cfg = local_path / "adapter_config.json"
        if adapter_cfg.exists():
            with open(adapter_cfg) as f:
                cfg = json.load(f)
            base_id = cfg.get("base_model_name_or_path", "")
            log.info("Detected LoRA adapter; base model: %s", base_id)
            base_model = AutoModelForCausalLM.from_pretrained(base_id)
            model = PeftModel.from_pretrained(base_model, str(local_path))
        else:
            model = AutoModelForCausalLM.from_pretrained(str(local_path))

        log.info("Pushing to Hub …")
        model.push_to_hub(repo_id, private=False)
        tokenizer.push_to_hub(repo_id, private=False)
    except Exception as e:
        log.warning("push_to_hub via transformers failed (%s); falling back to folder upload.", e)
        log.info("Uploading folder %s …", local_path)
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=repo_id,
            repo_type="model",
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload local model checkpoints to HuggingFace Hub.")

    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        help="Key from config/models.py to upload.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Upload all models whose hf_id is a local path.",
    )

    naming = p.add_mutually_exclusive_group(required=True)
    naming.add_argument(
        "--repo-prefix",
        metavar="ORG_OR_USER",
        help="HF namespace; repo name auto-derived from model name. E.g. 'myorg'.",
    )
    naming.add_argument(
        "--repo-id",
        metavar="ORG/REPO",
        help="Full repo id (only valid with --model, not --all).",
    )

    p.add_argument("--private", action="store_true", help="Create private repo.")
    p.add_argument(
        "--adapter-only",
        action="store_true",
        help="Only upload LoRA adapter files, not full model weights.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would happen without uploading.")
    p.add_argument("--token", default=None, help="HF token (defaults to HF_TOKEN env var or cached login).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Auth
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        log.info("Logged in via provided token.")
    elif not args.dry_run:
        log.info("No token provided; relying on cached HF credentials.")

    if args.repo_id and args.all:
        log.error("--repo-id cannot be used with --all (ambiguous). Use --repo-prefix instead.")
        sys.exit(1)

    # Build work list
    if args.all:
        targets = local_models()
        if not targets:
            log.warning("No models with local paths found in config/models.py.")
            sys.exit(0)
        log.info("Found %d local model(s): %s", len(targets), list(targets.keys()))
    else:
        cfg = MODELS[args.model]
        if not is_local_path(cfg.hf_id) and not args.dry_run:
            log.warning("Model '%s' has a non-local hf_id ('%s'). Nothing to upload.", args.model, cfg.hf_id)
            sys.exit(0)
        targets = {args.model: cfg}

    # Upload
    for key, cfg in targets.items():
        rid = args.repo_id if args.repo_id else repo_id_for(cfg, args.repo_prefix)
        upload_model(
            model_key=key,
            model_cfg=cfg,
            repo_id=rid,
            private=args.private,
            adapter_only=args.adapter_only,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

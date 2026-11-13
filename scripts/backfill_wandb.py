"""
Backfill a completed fine-tuning run into Weights & Biases.

The multilingual Safety-FT (base_sft_ml) was trained before wandb was wired in
(report_to="none"), but its per-step training metrics were captured in the run
log and its hyperparameters in training_info.json.  Rather than retrain (which
would change the adapter and invalidate the already-computed eval results), we
replay those captured metrics into a wandb run.

Usage:
    python scripts/backfill_wandb.py \
        --log /path/to/ml_pipeline.log \
        --info models/finetuned_ml/base_sft/training_info.json \
        --project cohere-safety-representations --name base_sft_ml
"""

import argparse
import ast
import json
import re
from pathlib import Path

LOSS_LINE = re.compile(r"\{'loss':.*\}")


def parse_metric_lines(log_path: Path) -> list[dict]:
    steps = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = LOSS_LINE.search(line)
        if not m:
            continue
        try:
            d = ast.literal_eval(m.group(0))
        except Exception:
            continue
        # Values were stringified by the logger; coerce numerics back to float.
        out = {}
        for k, v in d.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = v
        steps.append(out)
    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--info", required=True)
    parser.add_argument("--project", default="cohere-safety-representations")
    parser.add_argument("--name", default="base_sft_ml")
    parser.add_argument("--entity", default=None)
    args = parser.parse_args()

    import wandb

    info = json.loads(Path(args.info).read_text())
    steps = parse_metric_lines(Path(args.log))
    if not steps:
        raise SystemExit("No loss lines found in log — nothing to backfill.")

    config = {
        **info.get("hyperparams", {}),
        "method": info.get("method"),
        "data_source": info.get("data_source"),
        "num_examples": info.get("num_examples"),
        "base_model": info.get("base_model_hf_id"),
        "backfilled": True,
    }

    run = wandb.init(project=args.project, name=args.name, entity=args.entity,
                     config=config, job_type="sft",
                     notes="Backfilled from captured training log (multilingual Safety-FT).")
    for i, s in enumerate(steps):
        log = {f"train/{k}": v for k, v in s.items() if isinstance(v, (int, float))}
        step_key = int(round(s["epoch"] * 10)) if "epoch" in s else i
        run.log(log, step=step_key)

    summary = steps[-1]
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            run.summary[f"final/{k}"] = v
    run.finish()
    print(f"Backfilled {len(steps)} metric rows to wandb run '{args.name}' "
          f"(project={args.project}). URL above.")


if __name__ == "__main__":
    main()

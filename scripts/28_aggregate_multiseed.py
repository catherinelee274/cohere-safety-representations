"""Aggregate seed 42/43/44 behavioral and probe metrics as mean ± std."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_META = {
    "base_sft_ml_big": ("sft", "ml", 42),
    "base_sft_en3000": ("sft", "en3000", 42),
    "base_dpo_ml": ("dpo", "ml", 42),
    "base_dpo_en3000": ("dpo", "en3000", 42),
    "base_sft_ml_s43": ("sft", "ml", 43),
    "base_sft_en3000_s43": ("sft", "en3000", 43),
    "base_dpo_ml_s43": ("dpo", "ml", 43),
    "base_dpo_en3000_s43": ("dpo", "en3000", 43),
    "base_sft_ml_s44": ("sft", "ml", 44),
    "base_sft_en3000_s44": ("sft", "en3000", 44),
    "base_dpo_ml_s44": ("dpo", "ml", 44),
    "base_dpo_en3000_s44": ("dpo", "en3000", 44),
}


def add_meta(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["model"].isin(MODEL_META)].copy()
    frame[["stage", "arm", "seed"]] = frame["model"].apply(
        lambda model: pd.Series(MODEL_META[model])
    )
    return frame


def aggregate_behavioral() -> None:
    source = ROOT / "results/behavioral/baseline_summary.csv"
    frame = add_meta(pd.read_csv(source))
    output = ROOT / "results/behavioral/multiseed_summary.csv"
    summary = frame.groupby(["stage", "arm", "lang"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        refusal_rate_mean=("refusal_rate", "mean"),
        refusal_rate_std=("refusal_rate", "std"),
    )
    summary.to_csv(output, index=False)

    paired = frame.pivot_table(
        index=["stage", "lang", "seed"], columns="arm", values="refusal_rate"
    ).dropna().reset_index()
    paired["ml_minus_en3000"] = paired["ml"] - paired["en3000"]
    deltas = paired.groupby(["stage", "lang"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        difference_mean=("ml_minus_en3000", "mean"),
        difference_std=("ml_minus_en3000", "std"),
    )
    deltas.to_csv(ROOT / "results/behavioral/multiseed_differences.csv", index=False)
    print(f"Saved → {output}")


def aggregate_overrefusal() -> None:
    directory = ROOT / "results/behavioral"
    rows = []
    for model, (stage, arm, seed) in MODEL_META.items():
        for path in directory.glob(f"harmless_responses_{model}_*.jsonl"):
            import json
            lang = path.stem.rsplit("_", 1)[-1]
            with path.open() as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            rate = sum(record["heuristic_refuse"] for record in records) / max(len(records), 1)
            rows.append({
                "stage": stage, "arm": arm, "seed": seed, "lang": lang,
                "overrefusal_rate": rate,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    summary = frame.groupby(["stage", "arm", "lang"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        overrefusal_rate_mean=("overrefusal_rate", "mean"),
        overrefusal_rate_std=("overrefusal_rate", "std"),
    )
    output = directory / "multiseed_overrefusal_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Saved → {output}")


def aggregate_probes() -> None:
    source = ROOT / "results/probes/summary.csv"
    frame = add_meta(pd.read_csv(source))
    output = ROOT / "results/probes/multiseed_summary.csv"
    summary = frame.groupby(
        ["stage", "arm", "lang", "layer", "probe_type"], as_index=False
    ).agg(
        n_seeds=("seed", "nunique"),
        auc_mean=("auc", "mean"),
        auc_std=("auc", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    )
    summary.to_csv(output, index=False)
    print(f"Saved → {output}")


def main() -> None:
    aggregate_behavioral()
    aggregate_overrefusal()
    aggregate_probes()


if __name__ == "__main__":
    main()

"""Aggregate seeded ensemble-judge rates as mean ± std."""

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


def main() -> None:
    paths = [
        ROOT / "results/behavioral/judge_summary_ens.csv",
        ROOT / "results/behavioral/judge_summary_multiseed_ens.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["model"].isin(MODEL_META)].copy()
    frame = frame.sort_values("run_at").drop_duplicates(["model", "lang"], keep="last")
    frame[["stage", "arm", "seed"]] = frame["model"].apply(
        lambda model: pd.Series(MODEL_META[model])
    )
    summary = frame.groupby(["stage", "arm", "lang"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        unsafe_rate_mean=("unsafe_rate", "mean"),
        unsafe_rate_std=("unsafe_rate", "std"),
        kappa_mean=("interjudge_kappa", "mean"),
        kappa_std=("interjudge_kappa", "std"),
    )
    output = ROOT / "results/behavioral/multiseed_judge_summary.csv"
    summary.to_csv(output, index=False)
    print(f"Saved → {output}")


if __name__ == "__main__":
    main()

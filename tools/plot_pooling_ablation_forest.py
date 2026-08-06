#!/usr/bin/env python3
"""Create a three-panel paired-difference forest plot for pooling ablation."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

REFERENCE = "query_attention_8head"
COMPARATORS = ["masked_mean", "masked_max", "query_attention_1head"]
COMPARATOR_LABELS = {
    "masked_mean": "Mean",
    "masked_max": "Max",
    "query_attention_1head": "1-head",
}
METRICS = ["AUC", "AUPRC", "MCC"]
PANEL_LABELS = ["a", "b", "c"]
REQUIRED_OUTPUTS = (
    "pooling_ablation_forest.pdf",
    "pooling_ablation_forest.png",
    "pooling_ablation_forest_data.csv",
    "figure_caption.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/pooling_ablation/summaries/best_runs.csv"),
        help="Per-pooling, per-species best-run CSV.",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=Path("results/pooling_ablation/summaries/statistical_tests.csv"),
        help="Existing paired Wilcoxon/Holm results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pooling_ablation/figures"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def _atomic_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return final_path.parent / (
        f".{final_path.stem}.{uuid.uuid4().hex}{final_path.suffix}"
    )


def _atomic_csv(frame: pd.DataFrame, final_path: Path) -> None:
    temporary = _atomic_path(final_path)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(final_path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(text: str, final_path: Path) -> None:
    temporary = _atomic_path(final_path)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(final_path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_inputs(results: pd.DataFrame, statistics: pd.DataFrame) -> None:
    required_result_columns = {"pooling", "seed", "species_id", "species_name", *METRICS}
    missing = required_result_columns - set(results.columns)
    if missing:
        raise ValueError(f"Input results are missing columns: {sorted(missing)}")
    expected_poolings = {REFERENCE, *COMPARATORS}
    actual_poolings = set(results["pooling"])
    if actual_poolings != expected_poolings:
        raise ValueError(
            f"Expected pooling methods {sorted(expected_poolings)}, got {sorted(actual_poolings)}"
        )
    if set(results["seed"]) != {42}:
        raise ValueError(f"This completed experiment must contain only seed 42, got {sorted(results['seed'].unique())}")
    if results.duplicated(["pooling", "species_id"]).any():
        duplicates = results.loc[
            results.duplicated(["pooling", "species_id"], keep=False),
            ["pooling", "species_id"],
        ]
        raise ValueError(f"Duplicate pooling/species records found:\n{duplicates}")
    counts = results.groupby("pooling")["species_id"].nunique()
    if not (counts == 23).all():
        raise ValueError(f"Each pooling must contain 23 species, got {counts.to_dict()}")
    species_sets = {
        tuple(sorted(group["species_id"].tolist()))
        for _, group in results.groupby("pooling")
    }
    if len(species_sets) != 1:
        raise ValueError("Pooling methods do not contain identical species IDs")
    names_per_id = results.groupby("species_id")["species_name"].nunique()
    if not (names_per_id == 1).all():
        raise ValueError("Species names do not map one-to-one to species IDs")
    if results[["species_id", "species_name", *METRICS]].isna().any().any():
        raise ValueError("Input results contain missing species identifiers or metric values")

    required_stat_columns = {
        "metric", "reference", "competitor", "n_species", "p_raw", "p_holm"
    }
    missing_stats = required_stat_columns - set(statistics.columns)
    if missing_stats:
        raise ValueError(f"Statistics file is missing columns: {sorted(missing_stats)}")
    relevant = statistics[
        statistics["metric"].isin(METRICS)
        & statistics["competitor"].isin(COMPARATORS)
    ]
    if len(relevant) != 9 or relevant.duplicated(["metric", "competitor"]).any():
        raise ValueError("Statistics must contain exactly one row for each of the 9 comparisons")
    if set(relevant["reference"]) != {REFERENCE}:
        raise ValueError("Statistics comparison direction must use query_attention_8head as reference")
    if not (relevant["n_species"] == 23).all():
        raise ValueError("All statistical comparisons must use 23 species")
    if relevant[["p_raw", "p_holm"]].isna().any().any():
        raise ValueError("Statistics contain missing p-values")
    if not relevant["p_holm"].between(0, 1).all():
        raise ValueError("Holm-adjusted p-values must lie in [0, 1]")


def paired_forest_data(
    results: pd.DataFrame,
    statistics: pd.DataFrame,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if bootstrap_iterations < 20000:
        raise ValueError("--bootstrap-iterations must be at least 20000")
    validate_inputs(results, statistics)
    rng = np.random.default_rng(bootstrap_seed)
    indexed = results.set_index(["pooling", "species_id"]).sort_index()
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        reference = indexed.loc[REFERENCE, metric].sort_index()
        for comparator in COMPARATORS:
            comparison = indexed.loc[comparator, metric].sort_index()
            if not reference.index.equals(comparison.index):
                raise ValueError(f"Species alignment failed for {metric}/{comparator}")
            differences = (reference - comparison).to_numpy(dtype=float)
            sampled_means = rng.choice(
                differences,
                size=(bootstrap_iterations, len(differences)),
                replace=True,
            ).mean(axis=1)
            ci_lower, ci_upper = np.quantile(sampled_means, [0.025, 0.975])
            stat = statistics[
                (statistics["metric"] == metric)
                & (statistics["reference"] == REFERENCE)
                & (statistics["competitor"] == comparator)
            ]
            if len(stat) != 1:
                raise ValueError(f"Missing unique statistics row for {metric}/{comparator}")
            rows.append(
                {
                    "metric": metric,
                    "reference_method": REFERENCE,
                    "comparator": comparator,
                    "mean_difference": float(differences.mean()),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "p_raw": float(stat.iloc[0]["p_raw"]),
                    "p_holm": float(stat.iloc[0]["p_holm"]),
                    "n_species": len(differences),
                    "bootstrap_iterations": bootstrap_iterations,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 9:
        raise RuntimeError(f"Expected 9 forest rows, generated {len(frame)}")
    if not ((frame["ci_lower"] <= frame["mean_difference"]) &
            (frame["mean_difference"] <= frame["ci_upper"])).all():
        raise RuntimeError("A bootstrap confidence interval does not contain its mean")
    return frame


def draw_forest(frame: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    color = "#A51C30"
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.35), constrained_layout=False)
    y_positions = np.array([2, 1, 0])
    for panel_index, (ax, metric) in enumerate(zip(axes, METRICS)):
        panel = frame[frame["metric"] == metric].set_index("comparator").loc[COMPARATORS]
        means = panel["mean_difference"].to_numpy()
        lower = panel["ci_lower"].to_numpy()
        upper = panel["ci_upper"].to_numpy()
        xerr = np.vstack([means - lower, upper - means])
        max_abs = max(float(np.abs(np.r_[lower, upper]).max()), 0.001)
        limit = max_abs * 1.18
        ax.axvline(0, color="0.48", linestyle=(0, (3, 3)), linewidth=0.9, zorder=0)
        ax.errorbar(
            means,
            y_positions,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor=color,
            markersize=5.2,
            elinewidth=1.5,
            capsize=3.2,
            capthick=1.1,
            zorder=3,
        )
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.55, 2.55)
        ax.set_yticks(y_positions, [COMPARATOR_LABELS[item] for item in COMPARATORS])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel(f"Δ{metric} (8-head - comparator)")
        ax.set_title(metric, pad=8, weight="semibold")
        ax.text(-0.16, 1.08, PANEL_LABELS[panel_index], transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")
        for y, (_, row) in zip(y_positions, panel.iterrows()):
            star = " *" if row["p_holm"] < 0.05 else ""
            annotation = f"Δ={row['mean_difference']:+.4f}; p={row['p_holm']:.4f}{star}"
            ax.text(0.98, y + 0.22, annotation, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=7.1, color="0.18")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", direction="out", length=3)
        ax.set_facecolor("white")
    fig.text(0.5, 0.025, "Positive values favor 8-head attention", ha="center", va="bottom",
             fontsize=8.5, color="0.25")
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.23, top=0.86, wspace=0.60)
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 300})):
        final_path = output_dir / f"pooling_ablation_forest.{suffix}"
        temporary = _atomic_path(final_path)
        try:
            fig.savefig(temporary, format=suffix, facecolor="white", bbox_inches="tight", **kwargs)
            if temporary.stat().st_size == 0:
                raise RuntimeError(f"Generated empty {suffix.upper()} file")
            temporary.replace(final_path)
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def generate_forest_figure(
    input_path: Path,
    statistics_path: Path,
    output_dir: Path,
    bootstrap_iterations: int = 20000,
    bootstrap_seed: int = 42,
) -> pd.DataFrame:
    input_path = input_path.resolve()
    statistics_path = statistics_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input results file does not exist: {input_path}")
    if not statistics_path.is_file():
        raise FileNotFoundError(f"Statistics file does not exist: {statistics_path}")
    results = pd.read_csv(input_path)
    statistics = pd.read_csv(statistics_path)
    frame = paired_forest_data(
        results, statistics, bootstrap_iterations, bootstrap_seed
    )
    _atomic_csv(frame, output_dir / "pooling_ablation_forest_data.csv")
    caption = (
        "**图X｜不同池化方法相对于8头查询注意力池化的配对性能差值。** "
        "横坐标为8头查询注意力池化减去相应对比方法的配对性能差值，正值表示8头查询注意力池化表现更好。"
        "点表示23个物种配对差值的算术平均值，误差线表示以物种为抽样单位、采用percentile方法计算的95% bootstrap置信区间"
        f"（{bootstrap_iterations}次重复，随机种子{bootstrap_seed}）。竖直虚线表示差值为0。"
        "p值来自物种层面的双侧配对Wilcoxon signed-rank检验，并在同一指标的三次比较内使用Holm方法校正；"
        "图中报告校正后的p值。所有结果均来自随机种子42的一次训练。置信区间反映23个物种之间的差异，"
        "不表示随机种子波动。\n"
    )
    _atomic_text(caption, output_dir / "figure_caption.md")
    draw_forest(frame, output_dir)
    return frame


def main() -> int:
    args = parse_args()
    frame = generate_forest_figure(
        args.input,
        args.statistics,
        args.output_dir,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    print(f"Generated {len(frame)} paired forest estimates in {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

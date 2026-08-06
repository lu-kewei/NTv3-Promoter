#!/usr/bin/env python3
"""Summarize completed pooling-ablation runs and create publication figures."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

POOLINGS = ["masked_mean", "masked_max", "query_attention_1head", "query_attention_8head"]
METRICS = ["ACC", "Precision", "Recall", "F1", "AUC", "AUPRC", "MCC"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=Path("results/pooling_ablation"))
    parser.add_argument("--work-root", type=Path, default=Path("work_dirs/pooling_ablation"))
    parser.add_argument("--pooling", choices=POOLINGS, default=None, help="Validate one completed 69-run method.")
    parser.add_argument("--tie-threshold", type=float, default=0.005)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as f:
        frame.to_csv(f, index=False)
        temporary = Path(f.name)
    temporary.replace(path)


def collect(work_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows, epoch_rows = [], []
    for marker in sorted(work_root.glob("*/*/*/COMPLETE")):
        run_dir = marker.parent
        result_path = run_dir / "run_result.json"
        metrics_path = run_dir / "epoch_metrics.jsonl"
        if not result_path.is_file() or not metrics_path.is_file():
            raise RuntimeError(f"Incomplete completion marker in {run_dir}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        best_rows.append({key: result.get(key) for key in
                          ["pooling", "seed", "species_id", "species_name", "config", "run_dir",
                           "best_epoch", "best_checkpoint", *METRICS]})
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["MCC"] = row.pop("B_MCC", row.get("MCC"))
            row.update({key: result[key] for key in
                        ["pooling", "seed", "species_id", "species_name", "config", "run_dir"]})
            epoch_rows.append({key: row.get(key) for key in
                               ["pooling", "seed", "species_id", "species_name", "config", "run_dir",
                                "epoch", *METRICS, "loss"]})
    return pd.DataFrame(best_rows), pd.DataFrame(epoch_rows)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def make_plots(species: pd.DataFrame, figure_dir: Path, samples: int) -> None:
    if species.empty:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [item for item in POOLINGS if item in set(species.pooling)]
    rng = np.random.default_rng(20260805)
    plot_rows = []
    for metric in ("AUC", "AUPRC", "MCC"):
        for pooling in order:
            values = species.loc[species.pooling == pooling, f"{metric}_mean"].to_numpy(float)
            low, high = bootstrap_ci(values, samples, rng)
            plot_rows.append({"pooling": pooling, "metric": metric, "macro_mean": values.mean(),
                              "bootstrap_95_low": low, "bootstrap_95_high": high})
    atomic_csv(pd.DataFrame(plot_rows), figure_dir / "figure_data.csv")

    def draw(ax, metric):
        pivot = species.pivot(index="species_id", columns="pooling", values=f"{metric}_mean").reindex(columns=order)
        x = np.arange(len(order))
        for _, row in pivot.iterrows():
            ax.plot(x, row.values, color="0.72", lw=0.55, alpha=0.7, zorder=1)
            ax.scatter(x, row.values, s=10, color="0.3", alpha=0.6, zorder=2)
        stats = pd.DataFrame(plot_rows)
        stats = stats[stats.metric == metric].set_index("pooling").loc[order]
        y = stats.macro_mean.to_numpy()
        ax.errorbar(x, y, yerr=[y - stats.bootstrap_95_low, stats.bootstrap_95_high - y],
                    fmt="D", color="#b2182b", capsize=3, lw=1.3, label="Macro mean (95% bootstrap CI)")
        ax.set_xticks(x, order, rotation=18, ha="right")
        ax.set_ylabel(metric)
        ax.spines[["top", "right"]].set_visible(False)

    for metric in ("AUC", "AUPRC", "MCC"):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        draw(ax, metric)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            fig.savefig(figure_dir / f"pooling_ablation_{metric.lower()}.{suffix}", dpi=300)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric in zip(axes, ("AUC", "AUPRC", "MCC")):
        draw(ax, metric)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", frameon=False, ncol=1)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for suffix in ("pdf", "png"):
        fig.savefig(figure_dir / f"pooling_ablation_combined.{suffix}", dpi=300)
    plt.close(fig)
    (figure_dir / "figure_caption.md").write_text(
        "Points show each species mean across three seeds; grey lines pair species across pooling methods. "
        "Red diamonds show the unweighted 23-species macro mean and 95% species-level bootstrap confidence interval.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.tie_threshold < 0 or args.bootstrap_samples < 100:
        raise ValueError("tie threshold must be nonnegative and bootstrap samples must be >= 100")
    best, epochs = collect(args.work_root.resolve())
    if best.empty:
        raise RuntimeError(f"No completed runs found below {args.work_root}")
    if args.pooling:
        count = int((best.pooling == args.pooling).sum())
        manifest_path = args.result_root.resolve() / "manifests" / f"{args.pooling}.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Missing pooling manifest: {manifest_path}")
        expected = int(json.loads(manifest_path.read_text(encoding="utf-8"))["expected_runs"])
        if count != expected:
            raise RuntimeError(f"Expected {expected} completed {args.pooling} runs, found {count}")
    summaries = args.result_root.resolve() / "summaries"
    atomic_csv(epochs, summaries / "all_epochs.csv")
    atomic_csv(best, summaries / "best_runs.csv")

    aggregations = {metric: ["mean", "std"] for metric in METRICS}
    species = best.groupby(["pooling", "species_id", "species_name"], as_index=False).agg(aggregations)
    species.columns = ["_".join(part for part in col if part) if isinstance(col, tuple) else col for col in species.columns]
    atomic_csv(species, summaries / "species_seed_summary.csv")
    macro_rows = []
    for pooling, group in species.groupby("pooling"):
        row = {"pooling": pooling, "species_count": len(group)}
        for metric in METRICS:
            values = group[f"{metric}_mean"]
            row[f"{metric}_macro_mean"] = values.mean()
            row[f"{metric}_species_std"] = values.std(ddof=1)
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)
    atomic_csv(macro, summaries / "macro_summary.csv")

    reference = "query_attention_8head"
    win_rows, test_rows = [], []
    if reference in set(species.pooling):
        for competitor in [p for p in POOLINGS if p != reference and p in set(species.pooling)]:
            for metric in ("AUC", "AUPRC", "MCC"):
                paired = species[species.pooling.isin([reference, competitor])].pivot(
                    index="species_id", columns="pooling", values=f"{metric}_mean").dropna()
                delta = paired[competitor] - paired[reference]
                win_rows.append({"reference": reference, "competitor": competitor, "metric": metric,
                                 "tie_threshold": args.tie_threshold,
                                 "wins": int((delta > args.tie_threshold).sum()),
                                 "ties": int((delta.abs() <= args.tie_threshold).sum()),
                                 "losses": int((delta < -args.tie_threshold).sum())})
                if len(paired) == 23:
                    stat, p = wilcoxon(paired[reference], paired[competitor])
                    test_rows.append({"metric": metric, "reference": reference, "competitor": competitor,
                                      "n_species": 23, "statistic": stat, "p_raw": p})
    tests = pd.DataFrame(test_rows, columns=[
        "metric", "reference", "competitor", "n_species", "statistic", "p_raw"
    ])
    tests["p_holm"] = np.nan
    if not tests.empty:
        for metric, indices in tests.groupby("metric").groups.items():
            tests.loc[list(indices), "p_holm"] = holm(tests.loc[list(indices), "p_raw"].tolist())
    atomic_csv(pd.DataFrame(win_rows, columns=[
        "reference", "competitor", "metric", "tie_threshold", "wins", "ties", "losses"
    ]), summaries / "win_tie_loss.csv")
    atomic_csv(tests, summaries / "statistical_tests.csv")

    parameters = []
    for path in sorted(args.work_root.glob("*/*/*/parameter_counts.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pooling"] = path.parents[2].name
        parameters.append(payload)
    if parameters:
        atomic_csv(pd.DataFrame(parameters).drop_duplicates(), summaries / "parameter_counts.csv")
    efficiency_rows, efficiency_epoch_rows = [], []
    for result_path in sorted(args.work_root.glob("*/*/*/run_result.json")):
        run = json.loads(result_path.read_text(encoding="utf-8"))
        efficiency_path = result_path.parent / "training_efficiency_first50.json"
        if not efficiency_path.is_file():
            continue
        payload = json.loads(efficiency_path.read_text(encoding="utf-8"))
        identity = {key: run[key] for key in ("pooling", "seed", "species_id", "species_name", "run_dir")}
        efficiency_rows.append({**identity, **payload.get("summary", {})})
        for epoch_record in payload.get("epochs", []):
            efficiency_epoch_rows.append({**identity, **epoch_record})
    if efficiency_rows:
        atomic_csv(pd.DataFrame(efficiency_rows), summaries / "efficiency_summary.csv")
        atomic_csv(pd.DataFrame(efficiency_epoch_rows), summaries / "efficiency_epochs.csv")
    table = macro.copy()
    if not table.empty:
        table.insert(0, "Pooling parameters", table.pooling.map({
            "masked_mean": "0", "masked_max": "0", "query_attention_1head": "263,424",
            "query_attention_8head": "263,424"}))
        table = table.rename(columns={"pooling": "Pooling"})
        for metric in ("ACC", "F1", "AUC", "AUPRC", "MCC"):
            table[metric] = table.apply(lambda r: f"{r[f'{metric}_macro_mean']:.4f} ± {r[f'{metric}_species_std']:.4f}", axis=1)
        atomic_csv(table[["Pooling", "Pooling parameters", "ACC", "F1", "AUC", "AUPRC", "MCC"]],
                   summaries / "paper_table_pooling_ablation.csv")
    from plot_pooling_ablation_forest import generate_forest_figure

    generate_forest_figure(
        summaries / "best_runs.csv",
        summaries / "statistical_tests.csv",
        args.result_root.resolve() / "figures",
        bootstrap_iterations=max(20000, args.bootstrap_samples),
        bootstrap_seed=42,
    )
    print(f"Wrote summaries for {len(best)} completed runs to {summaries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

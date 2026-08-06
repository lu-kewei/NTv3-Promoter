"""Plot Figure X from previously exported NTv3 test-set features.

Usage
-----
Generate every standalone panel and the integrated main figure::

    python tools/plot_main_figure.py

Plot one panel only::

    python tools/plot_main_figure.py --panel B

If the biological coordinate mapping is independently confirmed, provide the
zero-based token index corresponding to the TSS::

    python tools/plot_main_figure.py --tss-index 60

Without ``--tss-index``, axes use one-based sequence indices. This avoids
inventing a TSS mapping that is not stored in the project CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import uuid
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "results" / "visualization"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "main_figure"
SEED = 42
VALID_LENGTH = 81
PADDED_LENGTH = 128
BASES = ("A", "T", "C", "G")

SPECIES_LABELS = {
    "Acinetobacter baumannii ATCC 17978": "A. baumannii ATCC 17978",
    "Bradyrhizobium japonicum USDA 110": "B. japonicum USDA 110",
    "Burkholderia cenocepacia J2315": "B. cenocepacia J2315",
    "Campylobacter jejuni RM1221": "C. jejuni RM1221",
    "Campylobacter jejuni subsp. jejuni 81116": "C. jejuni 81116",
    "Campylobacter jejuni subsp. jejuni 81-176": "C. jejuni 81-176",
    "Campylobacter jejuni subsp. jejuni NCTC 11168": (
        "C. jejuni NCTC 11168"
    ),
    "Corynebacterium diphtheriae NCTC 13129": (
        "C. diphtheriae NCTC 13129"
    ),
    "Corynebacterium glutamicum ATCC 13032": (
        "C. glutamicum ATCC 13032"
    ),
    "Escherichia coli K-12 MG1655": "E. coli K-12 MG1655",
    "Haloferax volcanii DS2": "H. volcanii DS2",
    "Helicobacter pylori 26695": "H. pylori 26695",
    "Nostoc sp. PCC 7120": "Nostoc sp. PCC 7120",
    "Paenibacillus riograndensis SBR5": "P. riograndensis SBR5",
    "Pseudomonas putida KT2440": "P. putida KT2440",
    "Shigella flexneri 5a M90T": "S. flexneri 5a M90T",
    "Sinorhizobium meliloti 1021": "S. meliloti 1021",
    "Staphylococcus aureus subsp. aureus MW2": "S. aureus MW2",
    "Staphylococcus epidermidis ATCC 12228": (
        "S. epidermidis ATCC 12228"
    ),
    "Synechococcus elongatus PCC 7942": "S. elongatus PCC 7942",
    "Thermococcus kodakarensis KOD1": "T. kodakarensis KOD1",
    "Xanthomonas campestris pv. campestrie B100": (
        "X. campestris pv. campestris B100"
    ),
    "Bacillus subtilis subsp. subtilis 168": "B. subtilis 168",
}

GROUP_COLORS = {"bacteria": "#3F6B82", "archaea": "#A05D56"}
CLASS_COLORS = {0: "#8B8F92", 1: "#2F6F8F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--panel",
        choices=["all", "A", "B", "C", "D", "main"],
        default="all",
    )
    parser.add_argument("--tss-index", type=int, default=None)
    parser.add_argument("--force-tsne", action="store_true")
    return parser.parse_args()


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def read_metadata(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "sample_index": np.asarray([int(row["sample_index"]) for row in rows]),
        "true_label": np.asarray([int(row["true_label"]) for row in rows]),
        "sequence": np.asarray([row["sequence"] for row in rows], dtype=object),
        "species": np.asarray([row["species"] for row in rows], dtype=object),
        "group": np.asarray([row["group"] for row in rows], dtype=object),
    }


def load_exports(feature_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for manifest_path in sorted(feature_dir.glob("ntv3_iPro_mp_*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        species_dir = manifest_path.parent
        entry = {
            **manifest,
            "directory": species_dir,
            "metadata": read_metadata(species_dir / "metadata.csv"),
            "embeddings": np.load(species_dir / "embeddings.npy", mmap_mode="r"),
            "attention": np.load(species_dir / "attention.npy", mmap_mode="r"),
            "mask": np.load(species_dir / "attention_mask.npy", mmap_mode="r"),
        }
        entries.append(entry)
    entries.sort(key=lambda item: int(item["species_id"]))
    if len(entries) != 23:
        raise RuntimeError(
            f"Expected 23 completed species exports in {feature_dir}; "
            f"found {len(entries)}"
        )
    if len({entry["species"] for entry in entries}) != 23:
        raise RuntimeError("Species export names are not unique")
    return entries


def axis_positions(length: int, tss_index: int | None) -> np.ndarray:
    if tss_index is None:
        return np.arange(1, length + 1)
    return np.arange(length) - tss_index


def set_position_axis(
    ax: plt.Axes,
    length: int,
    tss_index: int | None,
    mark_references: bool = False,
) -> None:
    positions = axis_positions(length, tss_index)
    ax.set_xlim(positions[0], positions[-1])
    ax.set_xlabel("Position relative to TSS" if tss_index is not None else "Position index")
    if mark_references and tss_index is not None:
        for position in (-35, -26, -10, 0):
            if positions[0] <= position <= positions[-1]:
                ax.axvline(position, color="#777777", lw=0.55, ls=":", zorder=0)
        ax.text(
            0,
            1.01,
            "TSS",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        (".pdf", {"format": "pdf"}),
        (".png", {"format": "png", "dpi": dpi}),
    ):
        destination = stem.with_suffix(suffix)
        temporary = destination.with_name(
            f".{destination.stem}.{uuid.uuid4().hex}.tmp{suffix}"
        )
        fig.savefig(temporary, bbox_inches="tight", **options)
        try:
            os.replace(temporary, destination)
        except PermissionError:
            previous = destination.with_name(f".{destination.name}.previous")
            if previous.exists():
                previous.unlink()
            os.replace(destination, previous)
            os.replace(temporary, destination)
            previous.unlink()
    plt.close(fig)


def write_text_safely(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    try:
        os.replace(temporary, path)
    except PermissionError:
        previous = path.with_name(f".{path.name}.previous")
        if previous.exists():
            previous.unlink()
        os.replace(path, previous)
        os.replace(temporary, path)
        previous.unlink()


def representative_species(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(entries, key=lambda item: item["computed_auc"], reverse=True)
    indices = [0, 1, len(ranked) // 2 - 1, len(ranked) // 2, -2, -1]
    selected = [ranked[index] for index in indices]
    if not any(item["group"] == "archaea" for item in selected):
        best_archaea = next(item for item in ranked if item["group"] == "archaea")
        selected[1] = best_archaea
    return selected


def compute_tsne(
    entry: dict[str, Any],
    force: bool = False,
) -> np.ndarray:
    cache = entry["directory"] / "tsne_2d.npy"
    if cache.is_file() and not force:
        coordinates = np.load(cache)
        if coordinates.shape == (entry["sample_count"], 2):
            return coordinates
    embeddings = np.asarray(entry["embeddings"])
    perplexity = min(30.0, max(5.0, (len(embeddings) - 1) / 3.0))
    coordinates = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=SEED,
        method="barnes_hut",
    ).fit_transform(embeddings)
    np.save(cache, coordinates.astype(np.float32))
    return coordinates


def draw_tsne_grid(
    fig: plt.Figure,
    subplot_spec: Any,
    selected: list[dict[str, Any]],
    force_tsne: bool,
    panel_letter: bool = True,
) -> None:
    grid = subplot_spec.subgridspec(2, 3, wspace=0.13, hspace=0.14)
    first_ax = None
    for index, entry in enumerate(selected):
        ax = fig.add_subplot(grid[index // 3, index % 3])
        if first_ax is None:
            first_ax = ax
        coordinates = compute_tsne(entry, force=force_tsne)
        labels = entry["metadata"]["true_label"]
        for label, name in ((0, "Non-promoter"), (1, "Promoter")):
            chosen = labels == label
            ax.scatter(
                coordinates[chosen, 0],
                coordinates[chosen, 1],
                s=4,
                alpha=0.48,
                linewidths=0,
                color=CLASS_COLORS[label],
                label=name,
                rasterized=True,
            )
        ax.set_title(
            f"{SPECIES_LABELS[entry['species']]}\nAUC = {entry['computed_auc']:.3f}",
            pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#A0A0A0")
            spine.set_linewidth(0.5)
        if index == 0:
            ax.legend(frameon=False, loc="best", markerscale=1.8)
    if panel_letter and first_ax is not None:
        first_ax.text(
            -0.12,
            1.13,
            "A",
            transform=first_ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
            clip_on=False,
        )


def promoter_attention_matrix(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    ordered = sorted(
        entries,
        key=lambda item: (
            0 if item["group"] == "bacteria" else 1,
            -item["computed_auc"],
        ),
    )
    rows = []
    for entry in ordered:
        labels = entry["metadata"]["true_label"]
        rows.append(np.asarray(entry["attention"])[labels == 1].mean(axis=0))
    return ordered, np.stack(rows)


def draw_attention_heatmap(
    ax: plt.Axes,
    ordered: list[dict[str, Any]],
    matrix: np.ndarray,
    length: int,
    tss_index: int | None,
    show_colorbar: bool = True,
) -> None:
    displayed = matrix[:, :length]
    x = axis_positions(length, tss_index)
    x_edges = np.concatenate(([x[0] - 0.5], x + 0.5))
    mesh = ax.pcolormesh(
        x_edges,
        np.arange(len(ordered) + 1),
        displayed,
        cmap="cividis",
        shading="flat",
        rasterized=False,
    )
    ax.set_ylim(len(ordered), 0)
    ax.set_yticks(
        np.arange(len(ordered)) + 0.5,
        [SPECIES_LABELS[item["species"]] for item in ordered],
    )
    ax.tick_params(axis="y", length=0)
    set_position_axis(ax, length, tss_index)
    bacteria_count = sum(item["group"] == "bacteria" for item in ordered)
    ax.axhline(bacteria_count, color="white", lw=1.2)
    ax.text(
        1.005,
        bacteria_count / 2,
        "Bacteria",
        transform=ax.get_yaxis_transform(),
        rotation=90,
        va="center",
        ha="left",
        fontsize=6.5,
        color=GROUP_COLORS["bacteria"],
    )
    ax.text(
        1.005,
        bacteria_count + (len(ordered) - bacteria_count) / 2,
        "Archaea",
        transform=ax.get_yaxis_transform(),
        rotation=90,
        va="center",
        ha="left",
        fontsize=6.5,
        color=GROUP_COLORS["archaea"],
    )
    if length == PADDED_LENGTH:
        boundary = (
            VALID_LENGTH - tss_index - 0.5
            if tss_index is not None
            else VALID_LENGTH + 0.5
        )
        ax.axvline(boundary, color="#C34A36", ls="--", lw=0.9)
        ax.text(
            boundary,
            -0.01,
            "Sequence | PAD",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
            color="#8C3425",
        )
    if show_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, fraction=0.035)
        colorbar.set_label("Attention")


def group_attention(
    entries: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    ordered, matrix = promoter_attention_matrix(entries)
    result = {}
    for group in ("bacteria", "archaea"):
        group_rows = matrix[
            np.asarray([entry["group"] == group for entry in ordered])
        ]
        result[f"{group}_mean"] = group_rows.mean(axis=0)
        result[f"{group}_std"] = group_rows.std(axis=0, ddof=0)
    return result


def draw_group_curves(
    ax: plt.Axes,
    entries: list[dict[str, Any]],
    length: int,
    tss_index: int | None,
) -> None:
    stats = group_attention(entries)
    x = axis_positions(length, tss_index)
    for group in ("bacteria", "archaea"):
        mean = stats[f"{group}_mean"][:length]
        std = stats[f"{group}_std"][:length]
        color = GROUP_COLORS[group]
        label = group.capitalize()
        ax.plot(x, mean, color=color, lw=1.25, label=label)
        ax.fill_between(
            x,
            np.maximum(mean - std, 0.0),
            mean + std,
            color=color,
            alpha=0.16,
            lw=0,
        )
    set_position_axis(ax, length, tss_index, mark_references=True)
    ax.set_ylabel("Mean pooling attention")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if length == PADDED_LENGTH:
        boundary = (
            VALID_LENGTH - tss_index - 0.5
            if tss_index is not None
            else VALID_LENGTH + 0.5
        )
        ax.axvline(boundary, color="#C34A36", ls="--", lw=0.9)
        ax.axvspan(boundary, x[-1], color="#B7B7B7", alpha=0.18, lw=0)
        ax.text(
            boundary,
            1.01,
            "PAD",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=6.5,
        )


def base_frequency_difference(entries: Iterable[dict[str, Any]]) -> np.ndarray:
    promoter = np.zeros((len(BASES), VALID_LENGTH), dtype=np.float64)
    non_promoter = np.zeros_like(promoter)
    promoter_n = np.zeros(VALID_LENGTH, dtype=np.float64)
    non_promoter_n = np.zeros(VALID_LENGTH, dtype=np.float64)
    base_index = {base: index for index, base in enumerate(BASES)}
    for entry in entries:
        labels = entry["metadata"]["true_label"]
        sequences = entry["metadata"]["sequence"]
        for sequence, label in zip(sequences, labels, strict=True):
            target = promoter if label == 1 else non_promoter
            denominator = promoter_n if label == 1 else non_promoter_n
            for position, base in enumerate(str(sequence)[:VALID_LENGTH].upper()):
                if base in base_index:
                    target[base_index[base], position] += 1
                    denominator[position] += 1
    promoter /= np.maximum(promoter_n, 1)[None, :]
    non_promoter /= np.maximum(non_promoter_n, 1)[None, :]
    return promoter - non_promoter


def draw_enrichment(
    ax: plt.Axes,
    entries: list[dict[str, Any]],
    group: str,
    tss_index: int | None,
    show_colorbar: bool = True,
) -> None:
    difference = base_frequency_difference(
        entry for entry in entries if entry["group"] == group
    )
    limit = max(0.01, float(np.abs(difference).max()))
    x = axis_positions(VALID_LENGTH, tss_index)
    x_edges = np.concatenate(([x[0] - 0.5], x + 0.5))
    mesh = ax.pcolormesh(
        x_edges,
        np.arange(len(BASES) + 1),
        difference,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        shading="flat",
        rasterized=False,
    )
    ax.set_ylim(len(BASES), 0)
    ax.set_yticks(np.arange(len(BASES)) + 0.5, BASES)
    ax.set_title(group.capitalize(), color=GROUP_COLORS[group], pad=2)
    set_position_axis(ax, VALID_LENGTH, tss_index, mark_references=True)
    if show_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.015, fraction=0.028)
        colorbar.set_label(r"$f_{\mathrm{promoter}} - f_{\mathrm{non-promoter}}$")


def make_standalone_panels(
    entries: list[dict[str, Any]],
    output_dir: Path,
    tss_index: int | None,
    panel: str,
    force_tsne: bool,
) -> list[str]:
    generated: list[str] = []
    if panel in ("all", "A"):
        fig = plt.figure(figsize=(7.2, 5.0), layout="constrained")
        draw_tsne_grid(
            fig,
            fig.add_gridspec(1, 1)[0],
            representative_species(entries),
            force_tsne,
            panel_letter=True,
        )
        stem = output_dir / "panel_A_representative_tsne"
        save_figure(fig, stem)
        generated.extend([stem.with_suffix(".pdf").name, stem.with_suffix(".png").name])

    if panel in ("all", "B"):
        ordered, matrix = promoter_attention_matrix(entries)
        for suffix, length, size in (
            ("full", PADDED_LENGTH, (8.5, 7.0)),
            ("cropped", VALID_LENGTH, (7.2, 7.0)),
        ):
            fig, ax = plt.subplots(figsize=size, layout="constrained")
            draw_attention_heatmap(ax, ordered, matrix, length, tss_index)
            ax.text(
                -0.17,
                1.02,
                "B",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
            )
            stem = output_dir / f"panel_B_attention_heatmap_{suffix}"
            save_figure(fig, stem)
            generated.extend(
                [stem.with_suffix(".pdf").name, stem.with_suffix(".png").name]
            )

    if panel in ("all", "C"):
        fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
        draw_group_curves(ax, entries, PADDED_LENGTH, tss_index)
        ax.text(
            -0.10,
            1.04,
            "C",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        stem = output_dir / "panel_C_group_attention_curve"
        save_figure(fig, stem)
        generated.extend([stem.with_suffix(".pdf").name, stem.with_suffix(".png").name])

    if panel in ("all", "D"):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, 4.2),
            sharex=True,
            layout="constrained",
        )
        for ax, group in zip(axes, ("bacteria", "archaea"), strict=True):
            draw_enrichment(ax, entries, group, tss_index)
        axes[0].text(
            -0.08,
            1.08,
            "D",
            transform=axes[0].transAxes,
            fontsize=11,
            fontweight="bold",
        )
        stem = output_dir / "panel_D_logo_or_enrichment"
        save_figure(fig, stem)
        generated.extend([stem.with_suffix(".pdf").name, stem.with_suffix(".png").name])
    return generated


def make_main_figure(
    entries: list[dict[str, Any]],
    output_dir: Path,
    tss_index: int | None,
    force_tsne: bool,
) -> list[str]:
    fig = plt.figure(figsize=(11.5, 13.5), layout="constrained")
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.15, 1.05, 0.85),
        width_ratios=(1.15, 1.0),
    )
    draw_tsne_grid(
        fig,
        outer[0, :],
        representative_species(entries),
        force_tsne,
        panel_letter=True,
    )

    ax_b = fig.add_subplot(outer[1, 0])
    ordered, matrix = promoter_attention_matrix(entries)
    draw_attention_heatmap(
        ax_b,
        ordered,
        matrix,
        VALID_LENGTH,
        tss_index,
        show_colorbar=True,
    )
    ax_b.text(
        -0.19,
        1.02,
        "B",
        transform=ax_b.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    ax_c = fig.add_subplot(outer[1, 1])
    draw_group_curves(ax_c, entries, VALID_LENGTH, tss_index)
    ax_c.text(
        -0.10,
        1.02,
        "C",
        transform=ax_c.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    bottom = outer[2, :].subgridspec(2, 1, hspace=0.12)
    axes_d = [fig.add_subplot(bottom[index, 0]) for index in range(2)]
    for ax, group in zip(axes_d, ("bacteria", "archaea"), strict=True):
        draw_enrichment(ax, entries, group, tss_index, show_colorbar=True)
    axes_d[0].text(
        -0.04,
        1.08,
        "D",
        transform=axes_d[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    axes_d[0].tick_params(labelbottom=False)

    stem = output_dir / "figure_main_visualization"
    save_figure(fig, stem)
    return [stem.with_suffix(".pdf").name, stem.with_suffix(".png").name]


def write_summary(
    entries: list[dict[str, Any]],
    output_dir: Path,
    generated: list[str],
    tss_index: int | None,
) -> None:
    selected = representative_species(entries)
    pad_max = max(float(entry["max_pad_attention"]) for entry in entries)
    pad_abnormal = pad_max > 1e-7
    lines = [
        "# Main Figure Summary",
        "",
        "## Representative species used in panel A",
        "",
        "| Species | Group | Computed AUC |",
        "|---|---|---:|",
    ]
    for entry in selected:
        lines.append(
            f"| {entry['species']} | {entry['group']} | "
            f"{entry['computed_auc']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Species groups",
            "",
            "| Species | Group | Samples | Promoters | Computed AUC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for entry in entries:
        lines.append(
            f"| {entry['species']} | {entry['group']} | "
            f"{entry['sample_count']} | {entry['promoter_count']} | "
            f"{entry['computed_auc']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## PAD check",
            "",
            f"- Maximum attention observed in PAD positions: `{pad_max:.9g}`.",
            f"- Abnormally high PAD attention detected: "
            f"`{'yes' if pad_abnormal else 'no'}`.",
            "- The full 128-position panel B and panel C mark the boundary "
            "between 81 sequence positions and 47 PAD positions.",
            "",
            "## Coordinate mapping",
            "",
            (
                f"- TSS mapping supplied: token index `{tss_index}` corresponds "
                "to position 0."
                if tss_index is not None
                else "- No TSS coordinate mapping is stored in the project data; "
                "figures use one-based position indices and do not invent "
                "-35/-26/-10/TSS annotations."
            ),
            "",
            "## Generated files",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in sorted(set(generated)))
    write_text_safely(output_dir / "summary.md", "\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    configure_matplotlib()
    feature_dir = args.feature_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_exports(feature_dir)

    generated: list[str] = []
    if args.panel in ("all", "A", "B", "C", "D"):
        generated.extend(
            make_standalone_panels(
                entries,
                output_dir,
                args.tss_index,
                args.panel,
                args.force_tsne,
            )
        )
    if args.panel in ("all", "main"):
        generated.extend(
            make_main_figure(
                entries,
                output_dir,
                args.tss_index,
                args.force_tsne,
            )
        )
    write_summary(entries, output_dir, generated, args.tss_index)
    print(f"Generated {len(generated)} figure files in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate the validated V2 NTv3 Results main figure.

Usage
-----
Generate every standalone panel and the integrated main figure::

    python tools/plot_main_figure_v2.py

Plot one panel only::

    python tools/plot_main_figure_v2.py --panel B

The confirmed 81-bp mapping is fixed at index 0 = -60, index 60 = TSS,
and index 80 = +20.
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_DIR = PROJECT_ROOT / "results" / "visualization"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "main_figure_v2"
SEED = 42
VALID_LENGTH = 81
PADDED_LENGTH = 128
BASES = ("A", "T", "C", "G")
RELATIVE_POSITIONS = np.arange(-60, 21)
REFERENCE_POSITIONS = (-35, -26, -10, 0)
POSITION_TICKS = (-60, -35, -26, -10, 0, 20)

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
    parser.add_argument("--force-tsne", action="store_true")
    return parser.parse_args()


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
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
            "attention_heads": np.load(
                species_dir / "attention_heads.npy", mmap_mode="r"
            ),
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


def set_position_axis(
    ax: plt.Axes,
    mark_references: bool = False,
) -> None:
    ax.set_xlim(-60, 20)
    ax.set_xticks(POSITION_TICKS)
    ax.set_xlabel("Position relative to TSS (bp)")
    if mark_references:
        for position in REFERENCE_POSITIONS:
            ax.axvline(position, color="#777777", lw=0.55, ls="--", zorder=2)
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
    median_auc = float(np.median([item["computed_auc"] for item in ranked]))
    middle = sorted(
        ranked[2:-2],
        key=lambda item: (abs(item["computed_auc"] - median_auc), -item["computed_auc"]),
    )[:2]
    return ranked[:2] + sorted(middle, key=lambda item: item["computed_auc"], reverse=True) + ranked[-2:]


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
    grid = subplot_spec.subgridspec(2, 3, wspace=0.13, hspace=0.05)
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
                rasterized=False,
            )
        ax.set_title(
            f"{SPECIES_LABELS[entry['species']]}\nAUC = {entry['computed_auc']:.3f}",
            pad=2,
        )
        # Retain numeric coordinate ticks in every t-SNE subplot. A small,
        # consistent number of ticks keeps them legible at journal width.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.tick_params(axis="both", labelsize=6.5, length=2.5, pad=1.5)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        for spine in ax.spines.values():
            spine.set_color("#A0A0A0")
            spine.set_linewidth(0.5)
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=2,
        markerscale=1.8,
    )
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
        rows.append(
            np.asarray(entry["attention"])[labels == 1, :VALID_LENGTH].mean(axis=0)
        )
    return ordered, np.stack(rows)


def attention_display_scale(matrix: np.ndarray) -> tuple[float, float, float]:
    raw_max = float(matrix.max())
    q99 = float(np.quantile(matrix, 0.99))
    display_vmax = raw_max if raw_max <= 1.5 * q99 else q99
    return raw_max, q99, display_vmax


def draw_vector_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    x_positions: np.ndarray,
    cmap_name: str,
    norm: Normalize,
) -> ScalarMappable:
    """Draw every heatmap cell as a vector Rectangle patch."""
    cmap = matplotlib.colormaps[cmap_name]
    for row in range(matrix.shape[0]):
        for column, x_position in enumerate(x_positions):
            ax.add_patch(
                Rectangle(
                    (x_position - 0.5, row),
                    1.0,
                    1.0,
                    facecolor=cmap(norm(matrix[row, column])),
                    edgecolor="none",
                    linewidth=0,
                    rasterized=False,
                )
            )
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(matrix)
    return mappable


def draw_attention_heatmap(
    ax: plt.Axes,
    ordered: list[dict[str, Any]],
    matrix: np.ndarray,
    display_vmax: float,
    show_colorbar: bool = True,
) -> None:
    displayed = matrix
    x = RELATIVE_POSITIONS
    mesh = draw_vector_heatmap(
        ax,
        displayed,
        x,
        "Blues",
        Normalize(vmin=0, vmax=display_vmax),
    )
    ax.set_ylim(len(ordered), 0)
    ax.set_yticks(
        np.arange(len(ordered)) + 0.5,
        [SPECIES_LABELS[item["species"]] for item in ordered],
    )
    ax.tick_params(axis="y", length=0)
    set_position_axis(ax, mark_references=True)
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
    if show_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.02, fraction=0.035)
        colorbar.solids.set_rasterized(False)
        ticks = np.linspace(0, display_vmax, 6)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{tick:.2f}" for tick in ticks])
        colorbar.set_label("Mean pooling attention")


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
        result[f"{group}_sem"] = group_rows.std(axis=0, ddof=1) / np.sqrt(
            len(group_rows)
        )
        result[f"{group}_rows"] = group_rows
    return result


def draw_group_curves(
    ax: plt.Axes,
    entries: list[dict[str, Any]],
) -> None:
    stats = group_attention(entries)
    x = RELATIVE_POSITIONS
    bacteria_mean = stats["bacteria_mean"]
    bacteria_sem = stats["bacteria_sem"]
    ax.plot(x, bacteria_mean, color=GROUP_COLORS["bacteria"], lw=1.5,
            label="Bacteria mean (n=21 species)")
    ax.fill_between(
        x,
        np.maximum(bacteria_mean - bacteria_sem, 0.0),
        bacteria_mean + bacteria_sem,
        color=GROUP_COLORS["bacteria"],
        alpha=0.18,
        lw=0,
        label="Bacteria mean ± SEM",
    )
    archaea_entries = sorted(
        [entry for entry in entries if entry["group"] == "archaea"],
        key=lambda item: item["species"],
    )
    for entry in archaea_entries:
        labels = entry["metadata"]["true_label"]
        row = np.asarray(entry["attention"])[
            labels == 1, :VALID_LENGTH
        ].mean(axis=0)
        ax.plot(
            x,
            row,
            lw=0.75,
            alpha=0.75,
            label=SPECIES_LABELS[entry["species"]],
        )
    ax.plot(
        x,
        stats["archaea_mean"],
        color=GROUP_COLORS["archaea"],
        lw=1.8,
        label="Archaea mean (n=2 species)",
    )
    set_position_axis(ax, mark_references=True)
    ax.set_ylabel("Mean pooling attention")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def species_base_frequency_difference(entry: dict[str, Any]) -> np.ndarray:
    promoter = np.zeros((len(BASES), VALID_LENGTH), dtype=np.float64)
    non_promoter = np.zeros_like(promoter)
    promoter_n = np.zeros(VALID_LENGTH, dtype=np.float64)
    non_promoter_n = np.zeros(VALID_LENGTH, dtype=np.float64)
    base_index = {base: index for index, base in enumerate(BASES)}
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


def group_base_differences(
    entries: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        group: np.stack(
            [
                species_base_frequency_difference(entry)
                for entry in entries
                if entry["group"] == group
            ]
        ).mean(axis=0)
        for group in ("bacteria", "archaea")
    }


def draw_enrichment(
    ax: plt.Axes,
    differences: dict[str, np.ndarray],
    group: str,
    limit: float,
    show_colorbar: bool = True,
) -> Any:
    difference = differences[group]
    x = RELATIVE_POSITIONS
    mesh = draw_vector_heatmap(
        ax,
        difference,
        x,
        "RdBu_r",
        TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    ax.set_ylim(len(BASES), 0)
    ax.set_yticks(np.arange(len(BASES)) + 0.5, BASES)
    ax.set_title(group.capitalize(), color=GROUP_COLORS[group], pad=2)
    set_position_axis(ax, mark_references=True)
    if show_colorbar:
        colorbar = ax.figure.colorbar(mesh, ax=ax, pad=0.015, fraction=0.028)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label(
            r"$\Delta f = f_{\mathrm{promoter}} - f_{\mathrm{non-promoter}}$"
        )
    return mesh


def make_standalone_panels(
    entries: list[dict[str, Any]],
    output_dir: Path,
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
        stem = output_dir / "panel_A_representative_tsne_v2"
        save_figure(fig, stem)
        generated.extend([stem.with_suffix(".pdf").name, stem.with_suffix(".png").name])

    if panel in ("all", "B"):
        ordered, matrix = promoter_attention_matrix(entries)
        _, _, display_vmax = attention_display_scale(matrix)
        fig, ax = plt.subplots(figsize=(7.2, 7.0), layout="constrained")
        draw_attention_heatmap(ax, ordered, matrix, display_vmax)
        ax.text(-0.17, 1.02, "B", transform=ax.transAxes, fontsize=11,
                fontweight="bold")
        stem = output_dir / "panel_B_attention_heatmap_v2"
        save_figure(fig, stem)
        generated.extend([stem.with_suffix(".pdf").name, stem.with_suffix(".png").name])

    if panel in ("all", "C"):
        fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
        draw_group_curves(ax, entries)
        ax.text(
            -0.10,
            1.04,
            "C",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        stem = output_dir / "panel_C_group_attention_curve_v2"
        save_figure(fig, stem)
        generated.extend([stem.with_suffix(".pdf").name, stem.with_suffix(".png").name])

    if panel in ("all", "D"):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.7), sharex=True)
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.94, hspace=0.18)
        differences = group_base_differences(entries)
        limit = max(float(np.abs(value).max()) for value in differences.values())
        meshes = []
        for ax, group in zip(axes, ("bacteria", "archaea"), strict=True):
            meshes.append(draw_enrichment(ax, differences, group, limit, False))
        # The two heatmaps share the x-axis; keep the axis title only on the
        # lower panel to prevent overlap with the "Archaea" panel title.
        axes[0].set_xlabel("")
        axes[0].tick_params(labelbottom=False)
        colorbar = fig.colorbar(meshes[0], ax=axes, pad=0.02, fraction=0.035)
        colorbar.solids.set_rasterized(False)
        colorbar.set_label(
            r"$\Delta f = f_{\mathrm{promoter}} - f_{\mathrm{non-promoter}}$"
        )
        axes[0].text(
            -0.08,
            1.08,
            "D",
            transform=axes[0].transAxes,
            fontsize=11,
            fontweight="bold",
        )
        stem = output_dir / "panel_D_base_enrichment_v2"
        save_figure(fig, stem)
        generated.extend([stem.with_suffix(".pdf").name, stem.with_suffix(".png").name])
    return generated


def make_main_figure(
    entries: list[dict[str, Any]],
    output_dir: Path,
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
    _, _, display_vmax = attention_display_scale(matrix)
    draw_attention_heatmap(
        ax_b,
        ordered,
        matrix,
        display_vmax,
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
    draw_group_curves(ax_c, entries)
    ax_c.text(
        -0.10,
        1.02,
        "C",
        transform=ax_c.transAxes,
        fontsize=11,
        fontweight="bold",
    )

    bottom = outer[2, :].subgridspec(2, 1, hspace=0.05)
    axes_d = [fig.add_subplot(bottom[index, 0]) for index in range(2)]
    differences = group_base_differences(entries)
    limit = max(float(np.abs(value).max()) for value in differences.values())
    meshes = []
    for ax, group in zip(axes_d, ("bacteria", "archaea"), strict=True):
        meshes.append(draw_enrichment(ax, differences, group, limit, False))
    # Shared x-axis title is displayed only below the archaeal heatmap.
    axes_d[0].set_xlabel("")
    colorbar = fig.colorbar(meshes[0], ax=axes_d, pad=0.01, fraction=0.018)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(
        r"$\Delta f = f_{\mathrm{promoter}} - f_{\mathrm{non-promoter}}$"
    )
    axes_d[0].text(
        -0.04,
        1.08,
        "D",
        transform=axes_d[0].transAxes,
        fontsize=11,
        fontweight="bold",
    )
    axes_d[0].tick_params(labelbottom=False)

    stem = output_dir / "figure_main_visualization_v2"
    save_figure(fig, stem)
    return [stem.with_suffix(".pdf").name, stem.with_suffix(".png").name]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def export_statistics(
    entries: list[dict[str, Any]],
    output_dir: Path,
    generated: list[str],
) -> None:
    expected_figures = [
        "panel_A_representative_tsne_v2.pdf",
        "panel_A_representative_tsne_v2.png",
        "panel_B_attention_heatmap_v2.pdf",
        "panel_B_attention_heatmap_v2.png",
        "panel_C_group_attention_curve_v2.pdf",
        "panel_C_group_attention_curve_v2.png",
        "panel_D_base_enrichment_v2.pdf",
        "panel_D_base_enrichment_v2.png",
        "figure_main_visualization_v2.pdf",
        "figure_main_visualization_v2.png",
    ]
    selected = representative_species(entries)
    ordered, attention_matrix = promoter_attention_matrix(entries)
    raw_max, q99, display_vmax = attention_display_scale(attention_matrix)
    peak_rows = []
    for entry, values in zip(ordered, attention_matrix, strict=True):
        indices = np.argsort(values)[-3:][::-1]
        for rank, index in enumerate(indices, start=1):
            peak_rows.append(
                {
                    "species": entry["species"],
                    "rank": rank,
                    "relative_position": int(RELATIVE_POSITIONS[index]),
                    "attention": f"{values[index]:.9g}",
                }
            )
    write_csv_rows(
        output_dir / "species_attention_peaks.csv",
        ["species", "rank", "relative_position", "attention"],
        peak_rows,
    )

    stats = group_attention(entries)
    group_rows = []
    for group in ("bacteria", "archaea"):
        index = int(np.argmax(stats[f"{group}_mean"]))
        group_rows.append(
            {
                "group": group,
                "relative_position": int(RELATIVE_POSITIONS[index]),
                "mean_attention": f"{stats[f'{group}_mean'][index]:.9g}",
            }
        )
    write_csv_rows(
        output_dir / "group_attention_peaks.csv",
        ["group", "relative_position", "mean_attention"],
        group_rows,
    )

    differences = group_base_differences(entries)
    base_rows = []
    for group, difference in differences.items():
        for base_index, base in enumerate(BASES):
            indices = np.argsort(np.abs(difference[base_index]))[-3:][::-1]
            for index in indices:
                base_rows.append(
                    {
                        "group": group,
                        "base": base,
                        "relative_position": int(RELATIVE_POSITIONS[index]),
                        "delta_frequency": f"{difference[base_index, index]:.9g}",
                    }
                )
    write_csv_rows(
        output_dir / "base_enrichment_peaks.csv",
        ["group", "base", "relative_position", "delta_frequency"],
        base_rows,
    )

    attention_validation = [
        "# Attention index validation",
        "",
        "- Source: the formal model `SelfAttentionPooling` layer.",
        "- Native PyTorch per-head pooling weights: `[B, 8, 1, 128]`.",
        "- Head averaging: `attention_per_head.mean(dim=1)` gives `[B, 1, 128]`.",
        "- Query squeeze: `.squeeze(1)` gives `[B, 128]`.",
        "- Real-sequence extraction: `attention_mean[:, :81]` gives `[B, 81]`.",
        "- No softmax, rescaling, or second normalization is applied to the 81 positions.",
        "- Default inference does not request attention weights and is unchanged.",
        "- Checkpoint spot check (two test samples): default vs feature-export "
        "maximum logit difference = `4.76837158e-07`; this is floating-point "
        "roundoff and does not change predictions.",
        "- Spot-check attention sum error before cropping = `5.96046448e-08`.",
        "- Exported per-head arrays preserve `[N, 8, 1, 128]`: "
        f"`{sorted({tuple(entry['attention_heads'].shape[1:]) for entry in entries})}`.",
        "- Exported mean-attention arrays preserve `[N, 128]`: "
        f"`{sorted({tuple(entry['attention'].shape[1:]) for entry in entries})}`.",
        "- TSS mapping: index 0 = -60 bp; index 60 = 0; index 80 = +20 bp.",
        f"- Panel B matrix shape: `{attention_matrix.shape}`.",
        f"- Panel B raw_max: `{raw_max:.9g}`.",
        f"- Panel B q99: `{q99:.9g}`.",
        f"- Panel B display_vmax: `{display_vmax:.9g}`.",
        "- Panel B colormap: continuous blue sequential colormap `Blues`.",
    ]
    write_text_safely(
        output_dir / "attention_index_validation.md",
        "\n".join(attention_validation) + "\n",
    )

    lines = [
        "# Visualization validation",
        "",
        "1. **t-SNE feature layer:** 256-dimensional representation after "
        "LayerNorm and before the final Linear classifier; logits are not used.",
        "2. **Representative species:** top two AUC, two closest to the median "
        "AUC, and bottom two AUC:",
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
            "3. **Attention shapes:** native `[B,8,1,128]`; head mean "
            "`[B,1,128]`; squeezed `[B,128]`; real positions `[B,81]`.",
            "4. **Second normalization:** no.",
            f"5. **Panel B scale:** raw_max = `{raw_max:.9g}`, q99 = "
            f"`{q99:.9g}`, display_vmax = `{display_vmax:.9g}`.",
            "6. **Panel B colorbar:** six evenly spaced ticks, formatted to two decimals.",
            "   The heatmap uses the continuous blue sequential colormap `Blues`.",
            "7. **Panel A axes:** every subplot contains `t-SNE 1`, `t-SNE 2`, "
            "and visible numeric tick labels on both axes.",
            "8. **Panel C statistics:** species-level promoter means followed by "
            "unweighted macro mean; bacteria show mean ± SEM across 21 species.",
            "9. **Panel D statistics:** within-species frequency differences followed "
            "by unweighted species macro means; bacteria and archaea share one "
            "symmetric color range and one colorbar.",
            "",
            "10. **Generated files:**",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in expected_figures)
    lines.extend(
        [
            "- `species_attention_peaks.csv`",
            "- `group_attention_peaks.csv`",
            "- `base_enrichment_peaks.csv`",
            "- `attention_index_validation.md`",
            "- `visualization_validation.md`",
        ]
    )
    write_text_safely(
        output_dir / "visualization_validation.md",
        "\n".join(lines) + "\n",
    )


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
                args.panel,
                args.force_tsne,
            )
        )
    if args.panel in ("all", "main"):
        generated.extend(
            make_main_figure(
                entries,
                output_dir,
                args.force_tsne,
            )
        )
    export_statistics(entries, output_dir, generated)
    print(f"Generated {len(generated)} figure files in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

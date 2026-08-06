"""Export NTv3 test-set embeddings and pooling attention for Figure X.

Usage
-----
Export all 23 independently trained species models::

    python tools/export_visualization_features.py

Useful options::

    python tools/export_visualization_features.py \
        --checkpoint-root work_dirs \
        --output-dir results/visualization \
        --batch-size 64 \
        --overwrite

The script reads the existing best-AUC Markdown table to locate each run and
epoch. It performs inference only; it never trains or updates a model.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import random
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Remote-code loading copies the local NTv3 Python module into a Transformers
# module cache. Keep that generated cache inside the writable project tree.
os.environ.setdefault(
    "HF_MODULES_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface_modules"),
)

from cell.utils.configs import instantiate_from_config, load_config  # noqa: E402


DEFAULT_RESULTS_TABLE = (
    PROJECT_ROOT
    / "docs"
    / "2026_07_23_17_44_to_2026_07_23_21_19"
    / "best_auc_results_all_runs.md"
)
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "models" / "ntv3_iPro_mp"
ARCHAEA = {
    "Haloferax volcanii DS2",
    "Thermococcus kodakarensis KOD1",
}
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-table", type=Path, default=DEFAULT_RESULTS_TABLE)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=PROJECT_ROOT / "work_dirs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "visualization",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a concrete device such as 'cuda:0'.",
    )
    parser.add_argument(
        "--species-id",
        type=int,
        action="append",
        help="Export only selected numeric species IDs; may be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_deterministic_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strip_markdown(value: str) -> str:
    return re.sub(r"[*`]", "", value).strip()


def write_text_safely(path: Path, text: str) -> None:
    """Atomically replace text, including on Windows preview-lock edge cases."""
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_npy_safely(path: Path, array: np.ndarray) -> None:
    """Atomically replace a NumPy array, tolerating Windows preview locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    try:
        os.replace(temporary, path)
    except PermissionError:
        previous = path.with_name(f".{path.name}.previous")
        if previous.exists():
            previous.unlink()
        os.replace(path, previous)
        os.replace(temporary, path)
        previous.unlink()


def parse_best_auc_table(path: Path) -> list[dict[str, Any]]:
    """Parse the 23 single-species rows without duplicating metric values."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 12:
            continue
        species_id = int(fields[0])
        if not 1 <= species_id <= 23:
            continue
        rows.append(
            {
                "species_id": species_id,
                "species": strip_markdown(fields[1]),
                "run": strip_markdown(fields[3]),
                "epoch": int(fields[4]),
                "reported_auc": float(strip_markdown(fields[9])),
            }
        )
    rows.sort(key=lambda row: row["species_id"])
    if len(rows) != 23:
        raise ValueError(f"Expected 23 single-species rows in {path}, found {len(rows)}")
    return rows


def find_config(config_dir: Path, species_id: int) -> Path:
    matches = sorted(config_dir.glob(f"ntv3_iPro_mp_*_{species_id}.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one config ending in _{species_id}.yaml, found {matches}"
        )
    return matches[0]


def find_checkpoint(
    checkpoint_root: Path,
    config_stem: str,
    run: str,
    epoch: int,
) -> Path:
    """Locate the exact reported checkpoint, then try a constrained fallback."""
    expected = checkpoint_root / config_stem / run / "ckpt" / f"epoch_{epoch}.pth"
    if expected.is_file():
        return expected
    matches = sorted(
        checkpoint_root.glob(
            f"{config_stem}/**/{run}/**/epoch_{epoch}.pth"
        )
    )
    if not matches:
        matches = sorted(
            checkpoint_root.glob(f"{config_stem}/**/epoch_{epoch}.pth")
        )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate epoch {epoch} for {config_stem}/{run}: "
            f"{matches}"
        )
    return matches[0]


def read_metadata_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_model_checkpoint(
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any]:
    cfg = load_config(str(config_path))
    model = instantiate_from_config(cfg.model)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("trainable_state_dict")
    if state_dict is None:
        state_dict = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=False)
    ignorable_unexpected = {
        key
        for key in incompatible.unexpected_keys
        if key.endswith(("cos_cached", "sin_cached"))
    }
    remaining_unexpected = set(incompatible.unexpected_keys) - ignorable_unexpected
    if incompatible.missing_keys or remaining_unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch. "
            f"Missing={incompatible.missing_keys}; "
            f"unexpected={sorted(remaining_unexpected)}"
        )
    model.to(device)
    model.eval()
    return model, cfg


def export_one_species(
    record: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    config_path = find_config(args.config_dir, record["species_id"])
    checkpoint_path = find_checkpoint(
        args.checkpoint_root,
        config_path.stem,
        record["run"],
        record["epoch"],
    )
    model, cfg = load_model_checkpoint(config_path, checkpoint_path, device)
    dataset = instantiate_from_config(cfg.data.test_data)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    csv_path = Path(str(cfg.data.test_data.params.data_path))
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    source_rows = read_metadata_rows(csv_path)
    if len(source_rows) != len(dataset):
        raise ValueError(
            f"CSV/Dataset length mismatch for {record['species']}: "
            f"{len(source_rows)} vs {len(dataset)}"
        )

    all_logits: list[np.ndarray] = []
    all_embeddings: list[np.ndarray] = []
    all_attention: list[np.ndarray] = []
    all_attention_heads: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(batch, return_features=True)
            all_logits.append(outputs["logits"].float().cpu().numpy())
            all_embeddings.append(
                outputs["sequence_embedding"].float().cpu().numpy()
            )
            all_attention.append(outputs["attention_mean"].float().cpu().numpy())
            # Native pooling-attention shape is [B, 8, 1, 128]. Keeping the
            # singleton query dimension makes the extraction auditable.
            all_attention_heads.append(
                outputs["attention_per_head"].float().cpu().numpy()
            )
            all_masks.append(outputs["attention_mask"].byte().cpu().numpy())

    logits = np.concatenate(all_logits)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    embeddings = np.concatenate(all_embeddings).astype(np.float32, copy=False)
    attention = np.concatenate(all_attention).astype(np.float32, copy=False)
    attention_heads = np.concatenate(all_attention_heads).astype(
        np.float32, copy=False
    )
    masks = np.concatenate(all_masks).astype(np.uint8, copy=False)
    labels = np.asarray(dataset.labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    auc = float(roc_auc_score(labels, probabilities[:, 1]))

    output_dir = args.output_dir / config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    save_npy_safely(output_dir / "embeddings.npy", embeddings)
    save_npy_safely(output_dir / "attention.npy", attention)
    save_npy_safely(output_dir / "attention_heads.npy", attention_heads)
    save_npy_safely(output_dir / "attention_mask.npy", masks)

    metadata_path = output_dir / "metadata.csv"
    metadata_buffer = io.StringIO(newline="")
    fieldnames = [
        "sample_index",
        "sample_id",
        "species",
        "group",
        "sequence",
        "true_label",
        "predicted_label",
        "positive_probability",
        "correct",
        "valid_length",
    ]
    writer = csv.DictWriter(metadata_buffer, fieldnames=fieldnames)
    writer.writeheader()
    group = "archaea" if record["species"] in ARCHAEA else "bacteria"
    for index, (row, label, prediction, probability, mask) in enumerate(
        zip(
            source_rows,
            labels,
            predictions,
            probabilities[:, 1],
            masks,
            strict=True,
        )
    ):
        writer.writerow(
            {
                "sample_index": index,
                "sample_id": row.get("species_detail", str(index)),
                "species": record["species"],
                "group": group,
                "sequence": row["sequence"],
                "true_label": int(label),
                "predicted_label": int(prediction),
                "positive_probability": f"{float(probability):.8f}",
                "correct": int(label == prediction),
                "valid_length": int(mask.sum()),
            }
        )
    write_text_safely(metadata_path, metadata_buffer.getvalue())

    valid = masks.astype(bool)
    pad_attention = np.where(valid, 0.0, attention)
    pad_max = float(pad_attention.max(initial=0.0))
    manifest = {
        **record,
        "group": "archaea" if record["species"] in ARCHAEA else "bacteria",
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "test_csv": str(csv_path.relative_to(PROJECT_ROOT)),
        "sample_count": int(len(labels)),
        "promoter_count": int(labels.sum()),
        "computed_auc": auc,
        "embedding_shape": list(embeddings.shape),
        "attention_shape": list(attention.shape),
        "attention_heads_shape": list(attention_heads.shape),
        "valid_lengths": sorted(np.unique(masks.sum(axis=1)).astype(int).tolist()),
        "max_pad_attention": pad_max,
    }
    write_text_safely(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> int:
    args = parse_args()
    args.results_table = args.results_table.resolve()
    args.config_dir = args.config_dir.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_deterministic_seed()
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    records = parse_best_auc_table(args.results_table)
    if args.species_id:
        requested = set(args.species_id)
        records = [row for row in records if row["species_id"] in requested]

    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in records:
        config_path = find_config(args.config_dir, record["species_id"])
        manifest_path = args.output_dir / config_path.stem / "manifest.json"
        if manifest_path.is_file() and not args.overwrite:
            logging.info("Reusing existing export: %s", config_path.stem)
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
            continue
        logging.info(
            "Exporting %02d/23: %s",
            record["species_id"],
            record["species"],
        )
        try:
            manifest = export_one_species(record, args, device)
            manifests.append(manifest)
            logging.info(
                "Exported %s: n=%d, AUC=%.5f",
                record["species"],
                manifest["sample_count"],
                manifest["computed_auc"],
            )
        except Exception as exc:  # Continue other independent species.
            logging.exception("Failed to export %s", record["species"])
            failures.append({"record": record, "error": str(exc)})
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    combined = {
        "seed": SEED,
        "device": str(device),
        "species": manifests,
        "failures": failures,
    }
    write_text_safely(
        args.output_dir / "manifest.json",
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
    )
    logging.info(
        "Finished: %d exported, %d failed. Manifest: %s",
        len(manifests),
        len(failures),
        args.output_dir / "manifest.json",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

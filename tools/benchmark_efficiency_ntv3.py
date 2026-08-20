"""Measure NTv3 runtime, process GPU memory, or forward FLOPs in isolation."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_MODULES_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface_modules"))

from cell.utils.configs import instantiate_from_config, load_config  # noqa: E402
from tools.efficiency_common import (  # noqa: E402
    BATCH_SIZE, MAX_LENGTH, NUM_WORKERS, RUNTIME_REPEATS,
    WARMUP_BATCHES, atomic_write_json, common_result, configure_device,
    environment_metadata, measure_memory, profile_forward, run_runtime_repeats,
)


DEFAULT_RESULTS_TABLE = (
    PROJECT_ROOT / "docs" / "2026_07_23_17_44_to_2026_07_23_21_19"
    / "best_auc_results_all_runs.md"
)
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs" / "models" / "ntv3_iPro_mp"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "efficiency" / "ntv3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("runtime", "memory", "flops"))
    parser.add_argument("--species-id", type=int, default=1, choices=range(1, 24))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--results-table", type=Path, default=DEFAULT_RESULTS_TABLE)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "work_dirs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=RUNTIME_REPEATS)
    parser.add_argument("--warmup-batches", type=int, default=WARMUP_BATCHES)
    parser.add_argument("--allow-non-4090", action="store_true", help="Smoke tests only.")
    args = parser.parse_args()
    if args.mode == "runtime" and args.repeats != RUNTIME_REPEATS:
        parser.error(f"formal runtime requires exactly {RUNTIME_REPEATS} repeats")
    if args.warmup_batches != WARMUP_BATCHES:
        parser.error(f"formal benchmark requires exactly {WARMUP_BATCHES} warm-up batches")
    if args.mode == "flops" and args.species_id != 1:
        parser.error("FLOPs uses species 1 as the architecture-representative checkpoint")
    return args


def _strip_markdown(value: str) -> str:
    return re.sub(r"[*`]", "", value).strip()


def checkpoint_record(table: Path, species_id: int) -> dict[str, Any]:
    records: dict[int, dict[str, Any]] = {}
    for line in table.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\|\s*\d+\s*\|", line):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if len(fields) != 12:
            continue
        row_id = int(fields[0])
        if 1 <= row_id <= 23:
            records[row_id] = {
                "species_id": row_id,
                "species_name": _strip_markdown(fields[1]),
                "run": _strip_markdown(fields[3]),
                "epoch": int(fields[4]),
            }
    if sorted(records) != list(range(1, 24)):
        raise ValueError(f"best-checkpoint table does not contain exactly species 1-23: {table}")
    return records[species_id]


def find_config(config_dir: Path, species_id: int) -> Path:
    matches = sorted(config_dir.glob(f"ntv3_iPro_mp_*_{species_id}.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one NTv3 config for species {species_id}, found {matches}")
    return matches[0]


def find_checkpoint(root: Path, config_stem: str, run: str, epoch: int) -> Path:
    expected = root / config_stem / run / "ckpt" / f"epoch_{epoch}.pth"
    if expected.is_file():
        return expected
    matches = sorted(root.glob(f"{config_stem}/**/epoch_{epoch}.pth"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not uniquely locate {config_stem} epoch {epoch}: {matches}")
    return matches[0]


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    cfg = load_config(str(config_path))
    if int(cfg.max_length_dna) != MAX_LENGTH:
        raise ValueError(f"NTv3 config max length is {cfg.max_length_dna}, expected {MAX_LENGTH}")
    model = instantiate_from_config(cfg.model)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("trainable_state_dict", payload.get("state_dict", payload))
    incompatible = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.endswith(("cos_cached", "sin_cached"))]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    if type(model).__name__ == "OptimizedModule" or hasattr(model, "_orig_mod"):
        raise RuntimeError("torch.compile models are forbidden in this benchmark")
    return model, cfg


def make_loader(cfg: Any) -> tuple[Any, DataLoader]:
    dataset = instantiate_from_config(cfg.data.test_data)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=False, drop_last=False,
    )
    return dataset, loader


def bf16_autocast_forward(
    model: torch.nn.Module, inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(inputs)


def warm_up(model: torch.nn.Module, loader: DataLoader, device: torch.device, count: int) -> None:
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= count:
                break
            inputs = {key: batch[key].to(device) for key in ("tokens", "attention_mask")}
            torch.softmax(bf16_autocast_forward(model, inputs), dim=1).cpu()
    torch.cuda.synchronize(device)


def prediction_function(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    expected = len(loader.dataset)

    def predict() -> list[int]:
        predictions: list[int] = []
        with torch.inference_mode():
            for batch in loader:
                inputs = {key: batch[key].to(device) for key in ("tokens", "attention_mask")}
                probabilities = torch.softmax(bf16_autocast_forward(model, inputs), dim=1)
                predictions.extend(probabilities.argmax(dim=1).cpu().tolist())
        if len(predictions) != expected:
            raise RuntimeError(f"NTv3 predicted {len(predictions)} samples, expected {expected}")
        return predictions
    return predict


def main() -> int:
    args = parse_args()
    device = configure_device(args.device, require_rtx4090=not args.allow_non_4090)
    record = checkpoint_record(args.results_table.resolve(), args.species_id)
    config_path = find_config(args.config_dir.resolve(), args.species_id)
    checkpoint_path = find_checkpoint(
        args.checkpoint_root.resolve(), config_path.stem, record["run"], record["epoch"]
    )
    model, cfg = load_model(config_path, checkpoint_path, device)
    dataset, loader = make_loader(cfg)
    if len(dataset) <= 0:
        raise RuntimeError("empty NTv3 test dataset")
    pipeline = {
        "checkpoint_selection": "best AUC checkpoint used by formal NTv3 evaluation",
        "config": str(config_path), "checkpoint": str(checkpoint_path),
        "best_run": record["run"], "best_epoch": record["epoch"],
        "best_table_species_name": record["species_name"],
        "model_count": 1, "online_tokenization": True,
        "precision_mode": "CUDA AMP autocast bfloat16; model tensors retain checkpoint dtypes",
        "prediction": "softmax(logits, dim=1), argmax, transfer predictions to CPU",
    }
    result = common_result(
        model_key="ntv3", mode=args.mode, species_id=args.species_id,
        num_samples=len(dataset), environment=environment_metadata(device), pipeline=pipeline,
    )
    predict = prediction_function(model, loader, device)

    if args.mode == "runtime":
        warm_up(model, loader, device, args.warmup_batches)
        runtime, digest = run_runtime_repeats(predict, device, args.repeats)
        result.update({
            "runtime": runtime,
            "runtime_boundary": (
                "after model/checkpoint load, eval, DataLoader creation and AMP warm-up; "
                "CUDA synchronized immediately before perf_counter and after complete-test-set prediction"
            ),
            "prediction_sha256": digest,
        })
        output = args.output_dir / f"species_{args.species_id:02d}_runtime.json"
    elif args.mode == "memory":
        warm_up(model, loader, device, args.warmup_batches)
        memory, digest = measure_memory(predict, device)
        result.update({"memory": memory, "prediction_sha256": digest})
        output = args.output_dir / f"species_{args.species_id:02d}_memory.json"
    else:
        sample = dataset[0]
        inputs = {
            "tokens": sample["tokens"].unsqueeze(0).to(device),
            "attention_mask": sample["attention_mask"].unsqueeze(0).to(device),
        }
        if tuple(inputs["tokens"].shape) != (1, MAX_LENGTH):
            raise RuntimeError(f"FLOPs input shape is {tuple(inputs['tokens'].shape)}, expected (1, 128)")
        profile_text = args.output_dir / "flops_profile.txt"
        result["flops"] = profile_forward(
            model, lambda: bf16_autocast_forward(model, inputs), profile_text
        )
        result["flops"]["input_ids_shape"] = list(inputs["tokens"].shape)
        output = args.output_dir / "flops.json"

    atomic_write_json(output.resolve(), result)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

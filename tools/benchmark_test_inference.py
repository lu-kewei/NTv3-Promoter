"""Benchmark end-to-end prediction on all 23 species test sets.

The benchmark locates the same best-AUC checkpoints used by the visualization
pipeline. Model/config/checkpoint loading and warm-up are excluded from timing.
Each measured repeat covers the complete test set and includes DataLoader work,
host-to-device transfer, model forward, argmax, and prediction transfer to CPU.

Examples
--------
Benchmark all species on GPU::

    python tools/benchmark_test_inference.py \
        --device cuda:0 \
        --mixed-precision bf16 \
        --batch-size 64 \
        --repeats 3

Smoke-test one species::

    python tools/benchmark_test_inference.py \
        --species-id 1 \
        --repeats 1 \
        --warmup-batches 1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import platform
import socket
import statistics
import sys
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import psutil
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# Keep generated Hugging Face remote-code modules inside the project tree.
os.environ.setdefault(
    "HF_MODULES_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface_modules"),
)

from cell.utils.configs import instantiate_from_config  # noqa: E402
from export_visualization_features import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    DEFAULT_RESULTS_TABLE,
    find_checkpoint,
    find_config,
    load_model_checkpoint,
    parse_best_auc_table,
    resolve_device,
    set_deterministic_seed,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "test_inference_benchmark"
JSON_FILENAME = "test_inference_benchmark.json"
CSV_FILENAME = "test_inference_benchmark.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-table", type=Path, default=DEFAULT_RESULTS_TABLE)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=PROJECT_ROOT / "work_dirs",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a concrete device such as 'cuda:0'.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="bf16",
        help="Autocast mode. CPU benchmarks always use 'no'.",
    )
    parser.add_argument(
        "--rss-sample-interval",
        type=float,
        default=0.01,
        help="Seconds between process-tree RSS samples.",
    )
    parser.add_argument(
        "--species-id",
        type=int,
        action="append",
        help="Benchmark only selected numeric species IDs; may be repeated.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches cannot be negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.rss_sample_interval <= 0:
        raise ValueError("--rss-sample-interval must be positive")
    if args.species_id:
        invalid = sorted(set(args.species_id) - set(range(1, 24)))
        if invalid:
            raise ValueError(f"Species IDs must be in 1--23; invalid: {invalid}")


def write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomically(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "species_id",
        "species",
        "sample_count",
        "batch_count",
        "batch_size",
        "repeat_count",
        "mean_prediction_time_seconds",
        "std_prediction_time_seconds",
        "min_prediction_time_seconds",
        "max_prediction_time_seconds",
        "mean_model_forward_time_seconds",
        "mean_seconds_per_sample",
        "mean_samples_per_second",
        "peak_gpu_allocated_gib",
        "peak_gpu_reserved_gib",
        "peak_process_tree_rss_gib",
        "peak_process_tree_rss_increase_gib",
        "predicted_negative_count",
        "predicted_positive_count",
        "checkpoint",
    ]
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    os.replace(temporary, path)


def process_tree_rss_bytes(process: psutil.Process) -> int:
    """Return current RSS for this process and its live descendants."""
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except (psutil.Error, OSError):
        pass
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.Error, OSError):
            continue
    return total


class PeakRssSampler:
    """Sample process-tree RSS in a background thread during one repeat."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.peak_bytes = max(
            self.peak_bytes,
            process_tree_rss_bytes(self.process),
        )

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self.baseline_bytes = process_tree_rss_bytes(self.process)
        self.peak_bytes = self.baseline_bytes
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="peak-rss-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_seconds))
        self._sample()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(
    device: torch.device,
    mixed_precision: str,
) -> Any:
    if device.type != "cuda" or mixed_precision == "no":
        return nullcontext()
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def prediction_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move only model inputs; labels are not required for prediction."""
    return {
        "tokens": batch["tokens"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(
            device,
            non_blocking=True,
        ),
    }


def warm_up(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    warmup_batches: int,
) -> None:
    if warmup_batches == 0:
        return
    with torch.inference_mode():
        for batch_index, batch in enumerate(data_loader):
            if batch_index >= warmup_batches:
                break
            inputs = prediction_batch(batch, device)
            with autocast_context(device, mixed_precision):
                outputs = model(inputs)
            # Materialize class predictions so warm-up follows the measured path.
            outputs.argmax(dim=-1).cpu()
    synchronize(device)


def make_cuda_event_pairs(
    count: int,
    device: torch.device,
) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
    if device.type != "cuda":
        return []
    return [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(count)
    ]


def measure_repeat(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    rss_sample_interval: float,
) -> dict[str, Any]:
    """Measure one complete deterministic traversal of the test set."""
    event_pairs = make_cuda_event_pairs(len(data_loader), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rss_sampler = PeakRssSampler(rss_sample_interval)
    predicted_counts = [0, 0]
    sample_count = 0
    cpu_model_forward_seconds = 0.0

    synchronize(device)
    rss_sampler.start()
    start_time = time.perf_counter()
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(data_loader):
                inputs = prediction_batch(batch, device)
                if device.type == "cuda":
                    start_event, end_event = event_pairs[batch_index]
                    start_event.record()
                    with autocast_context(device, mixed_precision):
                        outputs = model(inputs)
                    end_event.record()
                else:
                    model_start = time.perf_counter()
                    outputs = model(inputs)
                    cpu_model_forward_seconds += (
                        time.perf_counter() - model_start
                    )

                predictions = outputs.argmax(dim=-1).cpu()
                sample_count += int(predictions.numel())
                predicted_counts[0] += int((predictions == 0).sum().item())
                predicted_counts[1] += int((predictions == 1).sum().item())
        synchronize(device)
        end_time = time.perf_counter()
    finally:
        rss_sampler.stop()

    if device.type == "cuda":
        model_forward_seconds = sum(
            start_event.elapsed_time(end_event)
            for start_event, end_event in event_pairs
        ) / 1000.0
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        model_forward_seconds = cpu_model_forward_seconds
        peak_allocated = None
        peak_reserved = None

    return {
        "prediction_time_seconds": end_time - start_time,
        "model_forward_time_seconds": model_forward_seconds,
        "sample_count": sample_count,
        "predicted_negative_count": predicted_counts[0],
        "predicted_positive_count": predicted_counts[1],
        "peak_gpu_allocated_gib": (
            peak_allocated / (1024 ** 3)
            if peak_allocated is not None
            else None
        ),
        "peak_gpu_reserved_gib": (
            peak_reserved / (1024 ** 3)
            if peak_reserved is not None
            else None
        ),
        "rss_baseline_gib": rss_sampler.baseline_bytes / (1024 ** 3),
        "peak_process_tree_rss_gib": rss_sampler.peak_bytes / (1024 ** 3),
        "peak_process_tree_rss_increase_gib": max(
            0,
            rss_sampler.peak_bytes - rss_sampler.baseline_bytes,
        )
        / (1024 ** 3),
    }


def maximum_optional(values: Iterator[float | None]) -> float | None:
    available = [value for value in values if value is not None]
    return max(available) if available else None


def summarize_species(
    record: dict[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    dataset_size: int,
    batch_count: int,
    batch_size: int,
    repeats: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction_times = [
        repeat["prediction_time_seconds"] for repeat in repeats
    ]
    model_times = [
        repeat["model_forward_time_seconds"] for repeat in repeats
    ]
    mean_prediction_time = statistics.fmean(prediction_times)
    first_counts = (
        repeats[0]["predicted_negative_count"],
        repeats[0]["predicted_positive_count"],
    )
    if any(
        (
            repeat["predicted_negative_count"],
            repeat["predicted_positive_count"],
        )
        != first_counts
        for repeat in repeats[1:]
    ):
        raise RuntimeError(
            f"Predicted class counts changed across repeats for "
            f"{record['species']}"
        )
    if any(repeat["sample_count"] != dataset_size for repeat in repeats):
        raise RuntimeError(
            f"Prediction count does not match dataset size for "
            f"{record['species']}"
        )

    return {
        **record,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "sample_count": dataset_size,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "repeat_count": len(repeats),
        "mean_prediction_time_seconds": mean_prediction_time,
        "std_prediction_time_seconds": statistics.pstdev(prediction_times),
        "min_prediction_time_seconds": min(prediction_times),
        "max_prediction_time_seconds": max(prediction_times),
        "mean_model_forward_time_seconds": statistics.fmean(model_times),
        "mean_seconds_per_sample": mean_prediction_time / dataset_size,
        "mean_samples_per_second": dataset_size / mean_prediction_time,
        "peak_gpu_allocated_gib": maximum_optional(
            repeat["peak_gpu_allocated_gib"] for repeat in repeats
        ),
        "peak_gpu_reserved_gib": maximum_optional(
            repeat["peak_gpu_reserved_gib"] for repeat in repeats
        ),
        "peak_process_tree_rss_gib": max(
            repeat["peak_process_tree_rss_gib"] for repeat in repeats
        ),
        "peak_process_tree_rss_increase_gib": max(
            repeat["peak_process_tree_rss_increase_gib"]
            for repeat in repeats
        ),
        "predicted_negative_count": first_counts[0],
        "predicted_positive_count": first_counts[1],
        "repeats": repeats,
    }


def build_environment(
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    effective_precision = (
        args.mixed_precision if device.type == "cuda" else "no"
    )
    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None
    )
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "psutil_version": psutil.__version__,
        "device": str(device),
        "gpu_name": gpu_name,
        "mixed_precision_requested": args.mixed_precision,
        "mixed_precision_effective": effective_precision,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup_batches": args.warmup_batches,
        "repeats": args.repeats,
        "rss_sample_interval_seconds": args.rss_sample_interval,
    }


def build_document(
    args: argparse.Namespace,
    device: torch.device,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "measurement_scope": {
            "prediction_time": (
                "complete test-set DataLoader traversal + host-to-device "
                "transfer + model forward + argmax + prediction transfer to CPU"
            ),
            "excluded_from_prediction_time": (
                "model/config/checkpoint loading, dataset construction, "
                "warm-up, result serialization, and metric computation"
            ),
            "model_forward_time": (
                "sum of per-batch CUDA-event forward times on GPU; "
                "perf_counter forward time on CPU"
            ),
            "gpu_memory": (
                "PyTorch peak allocated/reserved memory during each measured "
                "repeat, including the already-loaded model"
            ),
            "cpu_memory": (
                "sampled RSS sum of the benchmark process and DataLoader "
                "children; peak may be underestimated between samples"
            ),
            "memory_unit": "GiB (bytes / 1024^3)",
        },
        "environment": build_environment(args, device),
        "species": results,
        "failures": failures,
    }


def persist_results(
    args: argparse.Namespace,
    device: torch.device,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    document = build_document(args, device, results, failures)
    write_text_atomically(
        args.output_dir / JSON_FILENAME,
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
    )
    write_csv_atomically(args.output_dir / CSV_FILENAME, results)


def benchmark_one_species(
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
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if len(dataset) == 0 or len(data_loader) == 0:
        raise ValueError(f"Empty test set for {record['species']}")

    effective_precision = (
        args.mixed_precision if device.type == "cuda" else "no"
    )
    warm_up(
        model,
        data_loader,
        device,
        effective_precision,
        args.warmup_batches,
    )

    repeat_records: list[dict[str, Any]] = []
    for repeat_index in range(1, args.repeats + 1):
        measured = measure_repeat(
            model,
            data_loader,
            device,
            effective_precision,
            args.rss_sample_interval,
        )
        measured["repeat"] = repeat_index
        repeat_records.append(measured)
        logging.info(
            "  repeat %d/%d: %.6f s, %.2f samples/s, "
            "GPU allocated=%s GiB, RSS=%.4f GiB",
            repeat_index,
            args.repeats,
            measured["prediction_time_seconds"],
            measured["sample_count"]
            / measured["prediction_time_seconds"],
            (
                f"{measured['peak_gpu_allocated_gib']:.4f}"
                if measured["peak_gpu_allocated_gib"] is not None
                else "unavailable"
            ),
            measured["peak_process_tree_rss_gib"],
        )

    result = summarize_species(
        record,
        config_path,
        checkpoint_path,
        len(dataset),
        len(data_loader),
        args.batch_size,
        repeat_records,
    )
    del data_loader
    del dataset
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    validate_args(args)
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
    if device.type == "cuda":
        torch.cuda.set_device(device)
    logging.info("Using device: %s", device)
    logging.info("Writing results to: %s", args.output_dir)

    records = parse_best_auc_table(args.results_table)
    if args.species_id:
        requested = set(args.species_id)
        records = [
            record
            for record in records
            if record["species_id"] in requested
        ]

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        logging.info(
            "Benchmarking %d/%d (species ID %02d): %s",
            index,
            len(records),
            record["species_id"],
            record["species"],
        )
        try:
            result = benchmark_one_species(record, args, device)
            results.append(result)
            logging.info(
                "Completed %s: mean %.6f s, %.2f samples/s",
                record["species"],
                result["mean_prediction_time_seconds"],
                result["mean_samples_per_second"],
            )
        except Exception as exc:  # Keep independent species auditable.
            logging.exception("Failed to benchmark %s", record["species"])
            failures.append({"record": record, "error": str(exc)})
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        persist_results(args, device, results, failures)

    logging.info(
        "Finished: %d succeeded, %d failed. JSON: %s; CSV: %s",
        len(results),
        len(failures),
        args.output_dir / JSON_FILENAME,
        args.output_dir / CSV_FILENAME,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared, model-agnostic utilities for the NTv3/iPro-MP efficiency study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIES_NAMES = [
    "Acinetobacter baumannii ATCC 17978",
    "Bradyrhizobium japonicum USDA 110",
    "Burkholderia cenocepacia J2315",
    "Campylobacter jejuni RM1221",
    "Campylobacter jejuni subsp. jejuni 81116",
    "Campylobacter jejuni subsp. jejuni 81-176",
    "Campylobacter jejuni subsp. jejuni NCTC 11168",
    "Corynebacterium diphtheriae NCTC 13129",
    "Corynebacterium glutamicum ATCC 13032",
    "Escherichia coli str K-12 substr. MG1655",
    "Haloferax volcanii DS2",
    "Helicobacter pylori strain 26695",
    "Nostoc sp. PCC7120",
    "Paenibacillus riograndensis SBR5",
    "Pseudomonas putida KT2440",
    "Shigella flexneri 5a str. M90T",
    "Sinorhizobium meliloti 1021",
    "Staphylococcus aureus subsp. aureus MW2",
    "Staphylococcus epidermidis ATCC 12228",
    "Synechococcus elongatus PCC 7942",
    "Thermococcus kodakarensis KOD1",
    "Xanthomonas campestris pv. campestrie B100",
    "Bacillus subtilis subsp. subtilis str. 168",
]

SCHEMA_VERSION = 1
DTYPE_NAME = "bfloat16"
BATCH_SIZE = 64
MAX_LENGTH = 128
NUM_WORKERS = 0
WARMUP_BATCHES = 5
RUNTIME_REPEATS = 30
TF32_ENABLED = False
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.02


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def configure_device(device_text: str, require_rtx4090: bool = True) -> torch.device:
    if not device_text.startswith("cuda:"):
        raise ValueError("formal efficiency experiments require an explicit CUDA device, e.g. cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no formal efficiency result was written")
    device = torch.device(device_text)
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    if require_rtx4090 and "RTX 4090" not in name:
        raise RuntimeError(f"formal experiment requires an RTX 4090, found {name!r}")
    torch.backends.cuda.matmul.allow_tf32 = TF32_ENABLED
    torch.backends.cudnn.allow_tf32 = TF32_ENABLED
    return device


def _nvidia_smi(arguments: list[str], timeout: float = 10.0) -> list[str]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", *arguments], check=False, capture_output=True,
            text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
    return completed.stdout.splitlines()


def environment_metadata(device: torch.device) -> dict[str, Any]:
    gpu_rows = _nvidia_smi([
        "--query-gpu=name,driver_version", "--format=csv,noheader,nounits"
    ])
    if not gpu_rows:
        raise RuntimeError("nvidia-smi returned no GPU metadata")
    fields = [item.strip() for item in gpu_rows[device.index or 0].rsplit(",", 1)]
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "driver_version": fields[1] if len(fields) == 2 else None,
        "cuda_runtime_version": torch.version.cuda,
        "pytorch_version": str(torch.__version__),
        "python_version": platform.python_version(),
        "dtype": DTYPE_NAME,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "num_dataloader_workers": NUM_WORKERS,
        "inference_mode": "torch.inference_mode",
        "torch_compile": False,
        "tf32_enabled": TF32_ENABLED,
        "cuda_device": str(device),
    }


def common_result(
    *, model_key: str, mode: str, species_id: int, num_samples: int,
    environment: dict[str, Any], pipeline: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_key,
        "mode": mode,
        "species_id": species_id,
        "species_name": SPECIES_NAMES[species_id - 1],
        "num_samples": num_samples,
        "environment": environment,
        "pipeline": pipeline,
    }


def prediction_digest(values: Iterable[int]) -> str:
    payload = bytes(int(value) for value in values)
    return hashlib.sha256(payload).hexdigest()


def runtime_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("runtime list is empty")
    return {
        "elapsed_times_s": values,
        "mean_s": statistics.fmean(values),
        "std_s": statistics.pstdev(values),
        "std_definition": "population standard deviation across complete-test-set repeats",
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def run_runtime_repeats(
    predict: Callable[[], list[int]], device: torch.device, repeats: int,
) -> tuple[dict[str, Any], str]:
    elapsed_times: list[float] = []
    digests: list[str] = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        start = time.perf_counter() # 开始计时
        predictions = predict()
        torch.cuda.synchronize(device)
        elapsed_times.append(time.perf_counter() - start) # 结束
        digests.append(prediction_digest(predictions))
    if len(set(digests)) != 1:
        raise RuntimeError("predictions changed across runtime repeats")
    return runtime_statistics(elapsed_times), digests[0]


def query_process_gpu_memory_mib(pid: int) -> int | None:
    rows = _nvidia_smi([
        "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"
    ])
    values: list[int] = []
    for line in rows:
        fields = [field.strip() for field in line.rsplit(",", 1)]
        if len(fields) != 2:
            continue
        try:
            if int(fields[0]) == pid:
                values.append(int(fields[1].split()[0]))
        except ValueError:
            continue
    return sum(values) if values else None


def stable_process_baseline_mib(
    pid: int, attempts: int = 20, pause_seconds: float = 0.1,
) -> tuple[int, list[int]]:
    readings: list[int] = []
    for _ in range(attempts):
        value = query_process_gpu_memory_mib(pid)
        if value is None:
            raise RuntimeError(f"nvidia-smi did not report benchmark PID {pid}")
        readings.append(value)
        if len(readings) >= 3 and max(readings[-3:]) - min(readings[-3:]) <= 1:
            return max(readings[-3:]), readings
        time.sleep(pause_seconds)
    raise RuntimeError(f"process GPU memory did not stabilize: {readings}")


class NvidiaSmiProcessSampler:
    """Continuously sample only the current benchmark PID's used GPU memory."""

    def __init__(self, pid: int, interval_seconds: float) -> None:
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples_mib: list[int] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def _read(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            fields = [field.strip() for field in line.rsplit(",", 1)]
            if len(fields) != 2:
                continue
            try:
                row_pid = int(fields[0])
                memory_mib = int(fields[1].split()[0])
            except ValueError:
                continue
            if row_pid == self.pid:
                self.samples_mib.append(memory_mib)

    def start(self) -> None:
        interval_ms = round(self.interval_seconds * 1000)
        command = [
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits", "-lms", str(interval_ms),
        ]
        try:
            self._process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"could not start nvidia-smi sampler: {exc}") from exc
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        time.sleep(max(0.05, 3 * self.interval_seconds))
        if self._process.poll() is not None:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"nvidia-smi sampler exited early: {stderr.strip()}")

    def sample_now(self) -> None:
        value = query_process_gpu_memory_mib(self.pid)
        if value is not None:
            self.samples_mib.append(value)

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("nvidia-smi sampling thread did not stop")


def measure_memory(
    predict: Callable[[], list[int]], device: torch.device,
) -> tuple[dict[str, Any], str]:
    torch.cuda.synchronize(device)
    pid = os.getpid()
    baseline_mib, baseline_readings = stable_process_baseline_mib(pid)
    torch.cuda.reset_peak_memory_stats(device)
    sampler = NvidiaSmiProcessSampler(pid, MEMORY_SAMPLE_INTERVAL_SECONDS)
    sampler.start()
    try:
        predictions = predict()
        torch.cuda.synchronize(device)
        sampler.sample_now()
    finally:
        sampler.stop()
    if not sampler.samples_mib:
        raise RuntimeError("no process GPU memory samples were captured")
    peak_mib = max([baseline_mib, *sampler.samples_mib])
    return ({
        "metric": "peak GPU process memory for current benchmark Python PID",
        "source": "nvidia-smi --query-compute-apps=pid,used_memory",
        "unit": "MiB",
        "sample_interval_seconds": MEMORY_SAMPLE_INTERVAL_SECONDS,
        "baseline_process_gpu_memory_mib": baseline_mib,
        "baseline_stability_readings_mib": baseline_readings,
        "peak_process_gpu_memory_mib": peak_mib,
        "peak_process_gpu_memory_gib": peak_mib / 1024.0,
        "peak_increase_mib": peak_mib - baseline_mib,
        "pytorch_peak_allocated_mib_audit": torch.cuda.max_memory_allocated(device) / 2**20,
        "pytorch_peak_reserved_mib_audit": torch.cuda.max_memory_reserved(device) / 2**20,
        "sample_count": len(sampler.samples_mib),
    }, prediction_digest(predictions))


def profile_forward(
    model: torch.nn.Module, forward: Callable[[], Any], profile_output: Path,
) -> dict[str, Any]:
    try:
        from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
    except Exception as exc:
        raise RuntimeError(
            "DeepSpeed Flops Profiler is required in both model environments"
        ) from exc
    version = importlib.metadata.version("deepspeed")
    profiler = FlopsProfiler(model)
    profiler.start_profile()
    try:
        with torch.inference_mode():
            forward()
        profiler.stop_profile()
        flops = int(profiler.get_total_flops(as_string=False))
        macs = int(profiler.get_total_macs(as_string=False))
        params = int(profiler.get_total_params(as_string=False))
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.print_model_profile(
            profile_step=1, module_depth=-1, top_modules=10, detailed=True,
            output_file=str(profile_output),
        )
        if flops <= 0 or macs <= 0 or params <= 0:
            raise RuntimeError(f"invalid profiler totals: FLOPs={flops}, MACs={macs}, params={params}")
        if not profile_output.is_file() or profile_output.stat().st_size == 0:
            raise RuntimeError(f"DeepSpeed did not create a detailed profile: {profile_output}")
        return {
            "profiler_name": "DeepSpeed Flops Profiler",
            "profiler_version": version,
            "profile_mode": "standalone inference forward",
            "batch_size": 1,
            "sequence_length": MAX_LENGTH,
            "flops_per_sample": flops,
            "gflops_per_sample": flops / 1e9,
            "macs_per_sample": macs,
            "gmacs_per_sample": macs / 1e9,
            "parameters": params,
            "flops_convention": (
                "DeepSpeed get_total_flops/get_total_macs counters are stored unchanged; "
                "the observed FLOPs/MAC ratio is recorded below"
            ),
            "observed_flops_per_mac": flops / macs,
            "coverage_audit": {
                "status": "passed",
                "rule": (
                    "DeepSpeed profiling completed without an exception, totals are positive, "
                    "and the detailed profile is non-empty"
                ),
                "unsupported_operation_policy": (
                    "DeepSpeed functional hooks are authoritative; a leaf module-level FLOPs "
                    "value of zero is not treated as unsupported because F.linear, layer_norm, "
                    "and similar functional calls may carry the operation count"
                ),
                "detailed_profile": str(profile_output),
                "detailed_profile_bytes": profile_output.stat().st_size,
            },
        }
    finally:
        profiler.end_profile()

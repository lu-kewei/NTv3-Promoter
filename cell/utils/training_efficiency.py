"""Training-loop efficiency measurement for the first 50 epochs."""

import json
import os
import os.path as osp
import statistics
import time
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator


class TrainingEfficiencyTracker:
    """Measure training-only time, throughput, and per-epoch GPU memory peaks.

    Epoch numbers are one-based in the generated report. Epochs 1--5 are
    recorded as warmup, epochs 6--50 are measured, and later epochs are
    ignored. All distributed processes participate in metric aggregation, but
    only the Accelerator main process writes JSON and emits efficiency logs.
    """

    MAX_RECORDED_EPOCH = 50
    WARMUP_EPOCHS = 5
    MEASURED_EPOCH_START = 6
    MEASURED_EPOCH_END = 50
    EXPECTED_MEASURED_EPOCH_COUNT = 45
    RESULT_FILENAME = "training_efficiency_first50.json"

    def __init__(
        self,
        accelerator: Accelerator,
        work_dir: str,
        logger: Any,
        batch_size: int,
        resume_existing: bool = False,
    ) -> None:
        """Initialize the tracker without changing model or optimizer state."""
        self.accelerator = accelerator
        self.logger = logger
        self.work_dir = osp.abspath(work_dir)
        self.result_path = osp.join(self.work_dir, self.RESULT_FILENAME)
        self.temp_result_path = self.result_path + ".tmp"
        self.device = accelerator.device
        self.gpu_memory_available = (
            torch.cuda.is_available() and self.device.type == "cuda"
        )
        self.environment = self._build_environment(batch_size)
        self.epoch_records: List[Dict[str, Any]] = self._load_existing_records(
            resume_existing
        )

        self._active = False
        self._paused = False
        self._start_time = 0.0
        self._pause_start_time = 0.0
        self._excluded_time = 0.0
        self._local_num_batches = 0
        self._local_num_samples = 0
        self._local_peak_allocated_bytes = 0
        self._local_peak_reserved_bytes = 0
        self._summary_logged = False

    def _build_environment(self, batch_size: int) -> Dict[str, Any]:
        """Return serializable runtime information for the result file."""
        gpu_name: Optional[str] = None
        gpu_count = 0
        if self.gpu_memory_available:
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_count = int(self.accelerator.num_processes)
        return {
            "device": str(self.device),
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "mixed_precision": str(self.accelerator.mixed_precision),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(
                self.accelerator.gradient_accumulation_steps
            ),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
        }

    def _load_existing_records(
        self, resume_existing: bool
    ) -> List[Dict[str, Any]]:
        """Load valid epoch records when resuming in the same work directory."""
        if (
            not resume_existing
            or not self.accelerator.is_main_process
            or not osp.isfile(self.result_path)
        ):
            return []
        try:
            with open(self.result_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            records = payload.get("epochs", [])
            if not isinstance(records, list):
                return []
            return [
                record
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("epoch"), int)
                and 1 <= record["epoch"] <= self.MAX_RECORDED_EPOCH
            ]
        except (OSError, ValueError, TypeError):
            self.logger.warning(
                "[Efficiency] Existing result file could not be read; "
                "starting a new measurement record."
            )
            return []

    def _synchronize_cuda(self) -> None:
        """Synchronize the local CUDA device when CUDA is in use."""
        if self.gpu_memory_available:
            torch.cuda.synchronize(self.device)

    def _reset_local_memory_peak(self) -> None:
        """Reset PyTorch peak-memory counters for the next training segment."""
        if self.gpu_memory_available:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _capture_local_memory_peak(self) -> None:
        """Accumulate the largest training-segment memory peaks this epoch."""
        if not self.gpu_memory_available:
            return
        self._local_peak_allocated_bytes = max(
            self._local_peak_allocated_bytes,
            int(torch.cuda.max_memory_allocated(self.device)),
        )
        self._local_peak_reserved_bytes = max(
            self._local_peak_reserved_bytes,
            int(torch.cuda.max_memory_reserved(self.device)),
        )

    def start_epoch(self, epoch_number: int) -> bool:
        """Start measuring a one-based epoch if it is within the first 50."""
        if not 1 <= epoch_number <= self.MAX_RECORDED_EPOCH:
            self._active = False
            return False

        self.accelerator.wait_for_everyone()
        self._synchronize_cuda()
        self._reset_local_memory_peak()

        self._excluded_time = 0.0
        self._local_num_batches = 0
        self._local_num_samples = 0
        self._local_peak_allocated_bytes = 0
        self._local_peak_reserved_bytes = 0
        self._paused = False
        self._active = True
        self._start_time = time.perf_counter()
        return True

    def record_batch(self, targets: Any) -> None:
        """Record one processed DataLoader batch and its actual local size."""
        if not self._active:
            return
        self._local_num_batches += 1
        if hasattr(targets, "shape") and len(targets.shape) > 0:
            self._local_num_samples += int(targets.shape[0])
        elif hasattr(targets, "__len__"):
            self._local_num_samples += int(len(targets))
        else:
            self._local_num_samples += 1

    def pause_for_excluded_work(self) -> None:
        """Pause timing before logging, validation, or other excluded work."""
        if not self._active or self._paused:
            return
        self._synchronize_cuda()
        self._pause_start_time = time.perf_counter()
        self._capture_local_memory_peak()
        self._paused = True

    def resume_after_excluded_work(self) -> None:
        """Resume timing after excluded work and start a new memory segment."""
        if not self._active or not self._paused:
            return
        self._synchronize_cuda()
        self._reset_local_memory_peak()
        self._excluded_time += time.perf_counter() - self._pause_start_time
        self._paused = False

    def finish_epoch(self, epoch_number: int) -> Optional[Dict[str, Any]]:
        """Finish, aggregate, persist, and log one completed epoch."""
        if not self._active:
            return None
        if self._paused:
            self.resume_after_excluded_work()

        self._synchronize_cuda()
        memory_capture_start = time.perf_counter()
        self._capture_local_memory_peak()
        self._excluded_time += time.perf_counter() - memory_capture_start
        self.accelerator.wait_for_everyone()
        self._synchronize_cuda()
        end_time = time.perf_counter()

        local_epoch_time = max(
            0.0, end_time - self._start_time - self._excluded_time
        )
        aggregated = self._aggregate_process_metrics(local_epoch_time)
        self._active = False

        if not self.accelerator.is_main_process:
            return None

        record = self._build_epoch_record(epoch_number, aggregated)
        self._replace_epoch_record(record)
        self._write_json_atomically()
        self._log_epoch(record)
        if epoch_number == self.MAX_RECORDED_EPOCH:
            self.log_summary()
        return record

    def _aggregate_process_metrics(self, local_epoch_time: float) -> Dict[str, Any]:
        """Aggregate local measurements and use the slowest/peak GPU values."""
        local_values = torch.tensor(
            [
                local_epoch_time,
                float(self._local_num_batches),
                float(self._local_num_samples),
                (
                    float(self._local_peak_allocated_bytes)
                    if self.gpu_memory_available
                    else -1.0
                ),
                (
                    float(self._local_peak_reserved_bytes)
                    if self.gpu_memory_available
                    else -1.0
                ),
            ],
            dtype=torch.float64,
            device=self.device,
        )
        gathered = self.accelerator.gather(local_values).reshape(-1, 5).cpu()
        allocated_bytes = float(gathered[:, 3].max().item())
        reserved_bytes = float(gathered[:, 4].max().item())
        return {
            "epoch_time_seconds": float(gathered[:, 0].max().item()),
            "num_batches": int(gathered[:, 1].max().item()),
            "num_samples": int(gathered[:, 2].sum().item()),
            "peak_gpu_allocated_gib": (
                allocated_bytes / (1024 ** 3) if allocated_bytes >= 0 else None
            ),
            "peak_gpu_reserved_gib": (
                reserved_bytes / (1024 ** 3) if reserved_bytes >= 0 else None
            ),
        }

    def _build_epoch_record(
        self, epoch_number: int, aggregated: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create one JSON epoch record from aggregated process metrics."""
        epoch_time = aggregated["epoch_time_seconds"]
        num_batches = aggregated["num_batches"]
        num_samples = aggregated["num_samples"]
        return {
            "epoch": epoch_number,
            "phase": (
                "warmup"
                if epoch_number <= self.WARMUP_EPOCHS
                else "measured"
            ),
            "epoch_time_seconds": epoch_time,
            "num_batches": num_batches,
            "num_samples": num_samples,
            "seconds_per_step": (
                epoch_time / num_batches if num_batches > 0 else 0.0
            ),
            "samples_per_second": (
                num_samples / epoch_time if epoch_time > 0 else 0.0
            ),
            "peak_gpu_allocated_gib": aggregated["peak_gpu_allocated_gib"],
            "peak_gpu_reserved_gib": aggregated["peak_gpu_reserved_gib"],
        }

    def _replace_epoch_record(self, record: Dict[str, Any]) -> None:
        """Insert or replace an epoch record and keep records sorted."""
        self.epoch_records = [
            existing
            for existing in self.epoch_records
            if existing.get("epoch") != record["epoch"]
        ]
        self.epoch_records.append(record)
        self.epoch_records.sort(key=lambda item: item["epoch"])

    def _build_summary(self) -> Dict[str, Any]:
        """Compute the epoch 6--50 summary with population standard deviation."""
        measured = [
            record
            for record in self.epoch_records
            if self.MEASURED_EPOCH_START
            <= record["epoch"]
            <= self.MEASURED_EPOCH_END
        ]
        expected_epochs = set(
            range(self.MEASURED_EPOCH_START, self.MEASURED_EPOCH_END + 1)
        )
        recorded_epochs = {record["epoch"] for record in measured}
        summary: Dict[str, Any] = {
            "completed_epoch_count": len(self.epoch_records),
            "measured_epoch_count": len(measured),
            "expected_measured_epoch_count": self.EXPECTED_MEASURED_EPOCH_COUNT,
            "measurement_complete": recorded_epochs == expected_epochs,
            "mean_epoch_time_seconds": None,
            "std_epoch_time_seconds": None,
            "min_epoch_time_seconds": None,
            "max_epoch_time_seconds": None,
            "mean_seconds_per_step": None,
            "mean_samples_per_second": None,
            "peak_gpu_allocated_gib": None,
            "peak_gpu_allocated_epoch": None,
            "peak_gpu_reserved_gib": None,
        }
        if not measured:
            return summary

        epoch_times = [record["epoch_time_seconds"] for record in measured]
        summary.update(
            {
                "mean_epoch_time_seconds": statistics.fmean(epoch_times),
                "std_epoch_time_seconds": statistics.pstdev(epoch_times),
                "min_epoch_time_seconds": min(epoch_times),
                "max_epoch_time_seconds": max(epoch_times),
                "mean_seconds_per_step": statistics.fmean(
                    record["seconds_per_step"] for record in measured
                ),
                "mean_samples_per_second": statistics.fmean(
                    record["samples_per_second"] for record in measured
                ),
            }
        )

        allocated_records = [
            record
            for record in measured
            if record["peak_gpu_allocated_gib"] is not None
        ]
        reserved_values = [
            record["peak_gpu_reserved_gib"]
            for record in measured
            if record["peak_gpu_reserved_gib"] is not None
        ]
        if allocated_records:
            peak_record = max(
                allocated_records,
                key=lambda record: record["peak_gpu_allocated_gib"],
            )
            summary["peak_gpu_allocated_gib"] = peak_record[
                "peak_gpu_allocated_gib"
            ]
            summary["peak_gpu_allocated_epoch"] = peak_record["epoch"]
        if reserved_values:
            summary["peak_gpu_reserved_gib"] = max(reserved_values)
        return summary

    def _build_document(self) -> Dict[str, Any]:
        """Build the complete JSON document."""
        return {
            "measurement_config": {
                "max_recorded_epoch": self.MAX_RECORDED_EPOCH,
                "warmup_epochs": self.WARMUP_EPOCHS,
                "measured_epoch_start": self.MEASURED_EPOCH_START,
                "measured_epoch_end": self.MEASURED_EPOCH_END,
                "expected_measured_epoch_count": (
                    self.EXPECTED_MEASURED_EPOCH_COUNT
                ),
                "memory_metric": "torch.cuda.max_memory_allocated",
                "memory_unit": "GiB",
                "time_scope": "training_loop_only",
                "std_definition": "population standard deviation (statistics.pstdev)",
                "distributed_epoch_time": "maximum across processes",
                "distributed_num_batches": "maximum across processes",
                "distributed_num_samples": "sum across processes",
                "distributed_gpu_memory": "maximum across processes",
            },
            "environment": self.environment,
            "epochs": self.epoch_records,
            "summary": self._build_summary(),
        }

    def _write_json_atomically(self) -> None:
        """Write through a temporary file and atomically replace the result."""
        if not self.accelerator.is_main_process:
            return
        os.makedirs(self.work_dir, exist_ok=True)
        with open(self.temp_result_path, "w", encoding="utf-8") as file:
            json.dump(
                self._build_document(),
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(self.temp_result_path, self.result_path)

    def _log_epoch(self, record: Dict[str, Any]) -> None:
        """Emit the requested one-line epoch efficiency message."""
        if record["peak_gpu_allocated_gib"] is None:
            memory_text = "peak_allocated=GPU memory unavailable"
        else:
            memory_text = "peak_allocated={:.4f} GiB".format(
                record["peak_gpu_allocated_gib"]
            )
        self.logger.info(
            "[Efficiency] epoch={} phase={} train_time={:.6f} s "
            "samples/s={:.6f} {}".format(
                record["epoch"],
                record["phase"],
                record["epoch_time_seconds"],
                record["samples_per_second"],
                memory_text,
            )
        )

    def log_summary(self) -> None:
        """Log the current summary at epoch 50 or when training terminates."""
        if not self.accelerator.is_main_process or self._summary_logged:
            return
        self._write_json_atomically()
        summary = self._build_summary()
        memory_value = summary["peak_gpu_allocated_gib"]
        memory_text = (
            "{:.4f} GiB".format(memory_value)
            if memory_value is not None
            else "GPU memory unavailable"
        )
        reserved_value = summary["peak_gpu_reserved_gib"]
        reserved_text = (
            "{:.4f} GiB".format(reserved_value)
            if reserved_value is not None
            else "GPU memory unavailable"
        )

        def format_optional(value: Optional[float], suffix: str = "") -> str:
            if value is None:
                return "unavailable"
            return "{:.6f}{}".format(value, suffix)

        self.logger.info(
            "\n[Efficiency Summary]\n"
            "Completed recorded epochs: {}\n"
            "Measured epochs: {} / {}\n"
            "Measurement complete: {}\n"
            "Mean epoch training time: {}\n"
            "Std epoch training time (population): {}\n"
            "Min epoch training time: {}\n"
            "Max epoch training time: {}\n"
            "Mean seconds per step: {}\n"
            "Mean samples per second: {}\n"
            "Peak GPU allocated memory: {}\n"
            "Peak GPU reserved memory: {}\n"
            "Peak memory epoch: {}\n"
            "Result file: {}".format(
                summary["completed_epoch_count"],
                summary["measured_epoch_count"],
                summary["expected_measured_epoch_count"],
                str(summary["measurement_complete"]).lower(),
                format_optional(summary["mean_epoch_time_seconds"], " s"),
                format_optional(summary["std_epoch_time_seconds"], " s"),
                format_optional(summary["min_epoch_time_seconds"], " s"),
                format_optional(summary["max_epoch_time_seconds"], " s"),
                format_optional(summary["mean_seconds_per_step"], " s"),
                format_optional(summary["mean_samples_per_second"]),
                memory_text,
                reserved_text,
                (
                    summary["peak_gpu_allocated_epoch"]
                    if summary["peak_gpu_allocated_epoch"] is not None
                    else "unavailable"
                ),
                self.result_path,
            )
        )
        self._summary_logged = True

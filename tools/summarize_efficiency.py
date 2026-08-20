"""Validate and summarize only the new NTv3 versus fixed-fold-1 iPro-MP results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "efficiency"
SPECIES_IDS = range(1, 24)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing formal efficiency result: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported or legacy result schema in {path}")
    return payload


def load_model_results(root: Path, model: str) -> dict[str, Any]:
    model_root = root / model
    runtime = [load_json(model_root / f"species_{sid:02d}_runtime.json") for sid in SPECIES_IDS]
    memory = [load_json(model_root / f"species_{sid:02d}_memory.json") for sid in SPECIES_IDS]
    flops = load_json(model_root / "flops.json")
    for mode, records in (("runtime", runtime), ("memory", memory)):
        for species_id, record in zip(SPECIES_IDS, records):
            if record.get("model") != model or record.get("mode") != mode:
                raise ValueError(f"model/mode mismatch in {model} species {species_id} {mode}")
            if record.get("species_id") != species_id:
                raise ValueError(f"species order mismatch in {model} {mode}")
    if flops.get("model") != model or flops.get("mode") != "flops":
        raise ValueError(f"model/mode mismatch in {model} FLOPs result")
    return {"runtime": runtime, "memory": memory, "flops": flops}


def validate_protocol(ntv3: dict[str, Any], ipro: dict[str, Any]) -> None:
    fixed_keys = (
        "gpu_name", "dtype", "batch_size", "max_length",
        "num_dataloader_workers", "inference_mode", "torch_compile", "tf32_enabled",
    )
    all_records = (
        ntv3["runtime"] + ntv3["memory"] + [ntv3["flops"]]
        + ipro["runtime"] + ipro["memory"] + [ipro["flops"]]
    )
    expected = {key: all_records[0]["environment"][key] for key in fixed_keys}
    for record in all_records[1:]:
        observed = {key: record["environment"][key] for key in fixed_keys}
        if observed != expected:
            raise ValueError(f"controlled protocol differs: expected {expected}, got {observed}")
    if expected != {
        "gpu_name": expected["gpu_name"], "dtype": "bfloat16", "batch_size": 64,
        "max_length": 128, "num_dataloader_workers": 0,
        "inference_mode": "torch.inference_mode", "torch_compile": False,
        "tf32_enabled": False,
    }:
        raise ValueError(f"formal controlled settings are invalid: {expected}")
    if "RTX 4090" not in expected["gpu_name"]:
        raise ValueError(f"formal results were not produced on RTX 4090: {expected['gpu_name']}")
    for sid in SPECIES_IDS:
        n_runtime = ntv3["runtime"][sid - 1]
        i_runtime = ipro["runtime"][sid - 1]
        n_memory = ntv3["memory"][sid - 1]
        i_memory = ipro["memory"][sid - 1]
        identity = (n_runtime["species_name"], n_runtime["num_samples"])
        for record in (i_runtime, n_memory, i_memory):
            if (record["species_name"], record["num_samples"]) != identity:
                raise ValueError(f"species/sample mismatch at ID {sid}")
        if n_runtime["prediction_sha256"] != n_memory["prediction_sha256"]:
            raise ValueError(f"NTv3 runtime/memory predictions differ at species {sid}")
        if i_runtime["prediction_sha256"] != i_memory["prediction_sha256"]:
            raise ValueError(f"iPro-MP runtime/memory predictions differ at species {sid}")
        if i_runtime["pipeline"].get("fold") != 1 or i_memory["pipeline"].get("fold") != 1:
            raise ValueError(f"iPro-MP species {sid} did not use fixed fold 1")
        for record in (n_memory, i_memory):
            if record["memory"].get("metric") != "peak GPU process memory for current benchmark Python PID":
                raise ValueError(f"wrong primary memory metric at species {sid}")
        if len(n_runtime["runtime"]["elapsed_times_s"]) != 30:
            raise ValueError(f"NTv3 species {sid} does not have 30 runtime repeats")
        if len(i_runtime["runtime"]["elapsed_times_s"]) != 30:
            raise ValueError(f"iPro-MP species {sid} does not have 30 runtime repeats")
    n_profiler = ntv3["flops"]["flops"]
    i_profiler = ipro["flops"]["flops"]
    if (n_profiler["profiler_name"], n_profiler["profiler_version"]) != (
        i_profiler["profiler_name"], i_profiler["profiler_version"]
    ):
        raise ValueError("FLOPs profiler name/version differs between the two models")
    if i_profiler.get("fold") != 1 or i_profiler.get("ensemble_multiplier") != 1:
        raise ValueError("iPro-MP FLOPs result is not fixed-fold-1 single-model FLOPs")
    if n_profiler["coverage_audit"].get("status") != "passed":
        raise ValueError("NTv3 FLOPs coverage audit did not pass")
    if i_profiler["coverage_audit"].get("status") != "passed":
        raise ValueError("iPro-MP FLOPs coverage audit did not pass")


def model_summary(results: dict[str, Any]) -> dict[str, Any]:
    species_means = [record["runtime"]["mean_s"] for record in results["runtime"]]
    peak_memories = [record["memory"]["peak_process_gpu_memory_mib"] for record in results["memory"]]
    flops = results["flops"]["flops"]
    return {
        "mean_runtime_across_23_species_s": statistics.fmean(species_means),
        "runtime_sd_across_species_s": statistics.pstdev(species_means),
        "runtime_sd_definition": "population SD of the 23 species-level mean runtimes",
        "maximum_peak_gpu_process_memory_mib": max(peak_memories),
        "maximum_peak_gpu_process_memory_gib": max(peak_memories) / 1024.0,
        "flops_per_sample": flops["flops_per_sample"],
        "gflops_per_sample": flops["gflops_per_sample"],
        "macs_per_sample": flops["macs_per_sample"],
        "gmacs_per_sample": flops["gmacs_per_sample"],
        "parameters": flops["parameters"],
        "profiler_name": flops["profiler_name"],
        "profiler_version": flops["profiler_version"],
        "flops_convention": flops["flops_convention"],
    }


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def render_markdown_report(
    rows: list[dict[str, Any]], summary: dict[str, Any],
    environment: dict[str, Any], smoke_status: str,
) -> str:
    ntv3 = summary["NTv3"]
    ipro = summary["iPro-MP"]
    runtime_ratio = (
        ipro["mean_runtime_across_23_species_s"]
        / ntv3["mean_runtime_across_23_species_s"]
    )
    runtime_reduction = (
        1
        - ntv3["mean_runtime_across_23_species_s"]
        / ipro["mean_runtime_across_23_species_s"]
    ) * 100
    memory_difference = (
        ipro["maximum_peak_gpu_process_memory_mib"]
        - ntv3["maximum_peak_gpu_process_memory_mib"]
    )
    memory_reduction = (
        memory_difference / ipro["maximum_peak_gpu_process_memory_mib"] * 100
    )
    flops_ratio = ipro["flops_per_sample"] / ntv3["flops_per_sample"]
    flops_reduction = (1 - ntv3["flops_per_sample"] / ipro["flops_per_sample"]) * 100
    parameter_ratio = ipro["parameters"] / ntv3["parameters"]

    lines = [
        "# NTv3 与 iPro-MP 推理效率实验结果",
        "",
        "> 对比对象：NTv3 正式 best checkpoint 与 iPro-MP 固定 fold 1 单模型。",
        "",
        "## 1. 主要结论",
        "",
        f"- NTv3 的23物种平均完整测试集推理时间为 "
        f"**{ntv3['mean_runtime_across_23_species_s']:.4f} s**，iPro-MP 为 "
        f"**{ipro['mean_runtime_across_23_species_s']:.4f} s**。按该口径，NTv3 "
        f"**快 {runtime_ratio:.2f} 倍**，平均时间减少 **{runtime_reduction:.2f}%**。",
        f"- NTv3 的最大进程峰值显存为 **{ntv3['maximum_peak_gpu_process_memory_mib']:.0f} MiB**，"
        f"iPro-MP 为 **{ipro['maximum_peak_gpu_process_memory_mib']:.0f} MiB**；NTv3 少用 "
        f"**{memory_difference:.0f} MiB（{memory_reduction:.2f}%）**。",
        f"- NTv3 每样本 forward 为 **{ntv3['gflops_per_sample']:.4f} GFLOPs**，"
        f"iPro-MP 为 **{ipro['gflops_per_sample']:.4f} GFLOPs**；iPro-MP 约为 NTv3 的 "
        f"**{flops_ratio:.2f} 倍**，NTv3 的 FLOPs 少 **{flops_reduction:.2f}%**。",
        f"- 参数量：NTv3 **{ntv3['parameters'] / 1e6:.3f} M**，iPro-MP "
        f"**{ipro['parameters'] / 1e6:.3f} M**，后者约为前者的 **{parameter_ratio:.2f} 倍**。",
        "",
        "## 2. 主结果表",
        "",
        "| 模型 | 23物种平均 runtime ± 物种间 SD (s) | 最大进程峰值显存 | GFLOPs/样本 | GMACs/样本 | 参数量 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| NTv3 | {ntv3['mean_runtime_across_23_species_s']:.4f} ± "
        f"{ntv3['runtime_sd_across_species_s']:.4f} | "
        f"{ntv3['maximum_peak_gpu_process_memory_mib']:.0f} MiB "
        f"({ntv3['maximum_peak_gpu_process_memory_gib']:.4f} GiB) | "
        f"{ntv3['gflops_per_sample']:.4f} | {ntv3['gmacs_per_sample']:.4f} | "
        f"{ntv3['parameters'] / 1e6:.3f} M |",
        f"| iPro-MP (fold 1) | {ipro['mean_runtime_across_23_species_s']:.4f} ± "
        f"{ipro['runtime_sd_across_species_s']:.4f} | "
        f"{ipro['maximum_peak_gpu_process_memory_mib']:.0f} MiB "
        f"({ipro['maximum_peak_gpu_process_memory_gib']:.4f} GiB) | "
        f"{ipro['gflops_per_sample']:.4f} | {ipro['gmacs_per_sample']:.4f} | "
        f"{ipro['parameters'] / 1e6:.3f} M |",
        "",
        "## 3. 逐物种结果",
        "",
        "表中 runtime 为每个物种30次完整测试集预测的平均值 ± 总体标准差；加速倍数定义为 iPro-MP / NTv3。",
        "",
        "| ID | 物种 | 样本数 | NTv3 runtime (s) | iPro-MP runtime (s) | NTv3 加速倍数 | NTv3 显存 (MiB) | iPro-MP 显存 (MiB) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['species_id']} | {row['species_name']} | {row['num_samples']} | "
            f"{row['ntv3_runtime_mean_s']:.4f} ± {row['ntv3_runtime_std_s']:.4f} | "
            f"{row['ipro_mp_runtime_mean_s']:.4f} ± {row['ipro_mp_runtime_std_s']:.4f} | "
            f"{row['runtime_speedup']:.2f}× | {row['ntv3_peak_gpu_memory_mib']:.0f} | "
            f"{row['ipro_mp_peak_gpu_memory_mib']:.0f} |"
        )
    lines.extend([
        "",
        "## 4. 实验环境",
        "",
        "| 项目 | 设置 |",
        "|---|---|",
        f"| GPU | {environment['gpu_name']} |",
        f"| NVIDIA driver | {environment['driver_version']} |",
        f"| PyTorch / CUDA | {environment['pytorch_version']} / {environment['cuda_runtime_version']} |",
        f"| Python | {environment['python_version']} |",
        f"| 精度 | BF16（NTv3 与 iPro-MP 均使用 CUDA AMP autocast） |",
        f"| Batch size / max length | {environment['batch_size']} / {environment['max_length']} |",
        f"| DataLoader workers | {environment['num_dataloader_workers']} |",
        f"| TF32 / torch.compile | {environment['tf32_enabled']} / {environment['torch_compile']} |",
        f"| FLOPs profiler | {ntv3['profiler_name']} {ntv3['profiler_version']} |",
        f"| iPro-MP smoke test | {smoke_status} |",
        "",
        "## 5. 口径说明",
        "",
        "- Runtime 主指标是23个物种各自 mean runtime 的算术宏平均，不是23个物种时间之和。",
        "- 主表中的 runtime SD 是23个物种 mean runtime 的总体标准差；逐物种表中的 SD 是该物种30次重复的总体标准差。",
        "- 显存是 `nvidia-smi` 当前 benchmark Python PID 的峰值进程显存，主指标取23个物种中的最大值。",
        "- FLOPs/MACs 是 batch size 1、长度128的单序列 inference forward；不是 FLOPS/s。",
        "- iPro-MP 使用每个物种固定 fold 1，未使用五折 ensemble。",
        "",
        "## 6. 机器可读结果",
        "",
        "- [逐物种 CSV](per_species.csv)",
        "- [总体 JSON](summary.json)",
        "- [NTv3 FLOPs 明细](../ntv3/flops_profile.txt)",
        "- [iPro-MP FLOPs 明细](../ipro_mp/flops_profile.txt)",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.result_root.resolve()
    ntv3 = load_model_results(root, "ntv3")
    ipro = load_model_results(root, "ipro_mp")
    validate_protocol(ntv3, ipro)
    rows: list[dict[str, Any]] = []
    for sid in SPECIES_IDS:
        n_runtime = ntv3["runtime"][sid - 1]
        i_runtime = ipro["runtime"][sid - 1]
        n_mean = n_runtime["runtime"]["mean_s"]
        i_mean = i_runtime["runtime"]["mean_s"]
        rows.append({
            "species_id": sid,
            "species_name": n_runtime["species_name"],
            "num_samples": n_runtime["num_samples"],
            "ntv3_runtime_mean_s": n_mean,
            "ntv3_runtime_std_s": n_runtime["runtime"]["std_s"],
            "ipro_mp_runtime_mean_s": i_mean,
            "ipro_mp_runtime_std_s": i_runtime["runtime"]["std_s"],
            "runtime_speedup": i_mean / n_mean,
            "runtime_speedup_definition": "iPro-MP mean runtime / NTv3 mean runtime",
            "ntv3_peak_gpu_memory_mib": ntv3["memory"][sid - 1]["memory"]["peak_process_gpu_memory_mib"],
            "ipro_mp_peak_gpu_memory_mib": ipro["memory"][sid - 1]["memory"]["peak_process_gpu_memory_mib"],
        })
    summary = {
        "schema_version": 1,
        "comparison": "NTv3 best checkpoint versus iPro-MP fixed fold 1",
        "species_count": 23,
        "runtime_primary_metric": "arithmetic macro mean of 23 species-level mean runtimes",
        "memory_primary_metric": "maximum peak current-PID GPU process memory across 23 species",
        "flops_metric": "forward FLOPs per one 128-token input sequence at batch size 1",
        "NTv3": model_summary(ntv3),
        "iPro-MP": model_summary(ipro),
    }
    output_dir = root / "summary"
    atomic_write_csv(output_dir / "per_species.csv", rows)
    atomic_write_json(output_dir / "summary.json", summary)
    smoke_path = root / "ipro_mp" / "smoke_test.json"
    smoke_status = "未找到 smoke_test.json"
    if smoke_path.is_file():
        smoke_status = load_json(smoke_path).get("smoke_test", {}).get("status", "未知")
    report = render_markdown_report(
        rows, summary, ntv3["runtime"][0]["environment"], smoke_status
    )
    atomic_write_text(output_dir / "efficiency_report.md", report)
    print(output_dir / "per_species.csv")
    print(output_dir / "summary.json")
    print(output_dir / "efficiency_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv

from tools.benchmark_efficiency_ipro_mp import dependency_compatibility, parse_fasta_sequences
from tools.benchmark_efficiency_ntv3 import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_RESULTS_TABLE,
    checkpoint_record,
    find_checkpoint,
    find_config,
)
from tools.efficiency_common import PROJECT_ROOT, SPECIES_NAMES, runtime_statistics
from tools.summarize_efficiency import model_summary, validate_protocol


def test_ntv3_has_one_resolvable_formal_best_checkpoint_per_species() -> None:
    for species_id in range(1, 24):
        record = checkpoint_record(DEFAULT_RESULTS_TABLE, species_id)
        config = find_config(DEFAULT_CONFIG_DIR, species_id)
        checkpoint = find_checkpoint(
            PROJECT_ROOT / "work_dirs", config.stem, record["run"], record["epoch"]
        )
        assert checkpoint.is_file()
        assert record["species_name"]


def test_fixed_fold1_checkpoint_exists_for_every_ipro_mp_species() -> None:
    model_dir = PROJECT_ROOT / "ipro-mp" / "models" / "07-final"
    expected = {f"{species_id}_fold_1.pth" for species_id in range(1, 24)}
    observed = {path.name for path in model_dir.glob("*_fold_1.pth")}
    assert observed == expected


def test_ipro_mp_required_transformers_apis_are_available() -> None:
    compatibility = dependency_compatibility()["packages"]
    assert compatibility["torch"]["version"]
    assert compatibility["transformers"]["version"]
    assert all(compatibility["transformers"]["api_check"].values())
    for optional in ("peft", "einops", "omegaconf", "accelerate", "evaluate"):
        assert compatibility[optional]["used_by_ipro_mp_benchmark"] is False


def test_two_pipelines_use_equal_test_sample_counts() -> None:
    config_dir = PROJECT_ROOT / "configs" / "models" / "ntv3_iPro_mp"
    fasta_dir = PROJECT_ROOT / "external" / "iPro-MP" / "Benchmark Dataset" / "Test"
    for species_id in range(1, 24):
        config_text = find_config(config_dir, species_id).read_text(encoding="utf-8")
        data_line = next(
            line for line in config_text.splitlines()
            if "data/IPro_MP/test/" in line
        )
        csv_path = PROJECT_ROOT / data_line.split("./", 1)[1].strip()
        with csv_path.open(newline="", encoding="utf-8") as handle:
            ntv3_count = sum(1 for _ in csv.DictReader(handle))
        ipro_count = len(parse_fasta_sequences(fasta_dir / f"{species_id}_test.txt"))
        assert ntv3_count == ipro_count


def test_runtime_statistics_keep_raw_repeats_and_use_population_sd() -> None:
    result = runtime_statistics([1.0, 2.0, 3.0])
    assert result["elapsed_times_s"] == [1.0, 2.0, 3.0]
    assert result["mean_s"] == 2.0
    assert result["median_s"] == 2.0
    assert result["min_s"] == 1.0
    assert result["max_s"] == 3.0
    assert result["std_definition"].startswith("population")


def test_ntv3_uses_amp_without_forcing_model_parameters_to_fp16() -> None:
    source = (
        PROJECT_ROOT / "tools" / "benchmark_efficiency_ntv3.py"
    ).read_text(encoding="utf-8")
    assert "model.to(device=device, dtype=torch.float16)" not in source
    assert "model.to(device=device, dtype=torch.bfloat16)" not in source
    assert ".half()" not in source
    assert ".bfloat16()" not in source
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source


def test_ipro_mp_uses_amp_without_forcing_model_parameter_dtype() -> None:
    source = (
        PROJECT_ROOT / "tools" / "benchmark_efficiency_ipro_mp.py"
    ).read_text(encoding="utf-8")
    assert ".half()" not in source
    assert ".bfloat16()" not in source
    assert "model.to(device=device, dtype=torch.float16)" not in source
    assert "model.to(device=device, dtype=torch.bfloat16)" not in source
    assert "model.to(device)" in source
    assert 'torch.autocast(device_type="cuda", dtype=torch.bfloat16)' in source


def test_bf16_benchmark_numpy_boundaries_cast_outputs_to_float32() -> None:
    for benchmark in (
        "benchmark_efficiency_ntv3.py",
        "benchmark_efficiency_ipro_mp.py",
    ):
        source = (PROJECT_ROOT / "tools" / benchmark).read_text(encoding="utf-8")
        numpy_call_count = source.count(".numpy()")
        safe_numpy_call_count = source.count(".cpu().float().numpy()")
        assert safe_numpy_call_count == numpy_call_count


def test_flops_audit_does_not_treat_zero_leaf_module_flops_as_unsupported() -> None:
    source = (
        PROJECT_ROOT / "tools" / "efficiency_common.py"
    ).read_text(encoding="utf-8")
    assert "possible unsupported parameterized leaf operations" not in source
    assert "zero_flop_parameter_leaves" not in source
    assert "DeepSpeed profiling completed without an exception" in source
    assert "detailed profile is non-empty" in source


def _synthetic_model_results(model: str):
    environment = {
        "gpu_name": "NVIDIA GeForce RTX 4090", "dtype": "bfloat16",
        "batch_size": 64, "max_length": 128, "num_dataloader_workers": 0,
        "inference_mode": "torch.inference_mode", "torch_compile": False,
        "tf32_enabled": False,
    }
    runtime, memory = [], []
    for species_id, species_name in enumerate(SPECIES_NAMES, 1):
        pipeline = {"fold": 1} if model == "ipro_mp" else {}
        runtime.append({
            "species_id": species_id, "species_name": species_name, "num_samples": 10,
            "environment": environment, "pipeline": pipeline,
            "prediction_sha256": "same", "runtime": {
                "elapsed_times_s": [1.0] * 30, "mean_s": 1.0, "std_s": 0.0,
            },
        })
        memory.append({
            "species_id": species_id, "species_name": species_name, "num_samples": 10,
            "environment": environment, "pipeline": pipeline,
            "prediction_sha256": "same", "memory": {
                "metric": "peak GPU process memory for current benchmark Python PID",
                "peak_process_gpu_memory_mib": 100,
            },
        })
    flops_data = {
        "profiler_name": "DeepSpeed Flops Profiler", "profiler_version": "test",
        "coverage_audit": {"status": "passed"}, "flops_per_sample": 20,
        "gflops_per_sample": 2e-8, "macs_per_sample": 10,
        "gmacs_per_sample": 1e-8, "parameters": 5, "flops_convention": "test",
    }
    if model == "ipro_mp":
        flops_data.update({"fold": 1, "ensemble_multiplier": 1})
    return {
        "runtime": runtime, "memory": memory,
        "flops": {"environment": environment, "flops": flops_data},
    }


def test_summary_accepts_only_the_controlled_fixed_fold1_protocol() -> None:
    ntv3 = _synthetic_model_results("ntv3")
    ipro = _synthetic_model_results("ipro_mp")
    validate_protocol(ntv3, ipro)
    assert model_summary(ntv3)["mean_runtime_across_23_species_s"] == 1.0
    ipro["runtime"][0]["pipeline"]["fold"] = 2
    try:
        validate_protocol(ntv3, ipro)
    except ValueError as exc:
        assert "fold 1" in str(exc)
    else:
        raise AssertionError("summary accepted a non-fold-1 iPro-MP result")

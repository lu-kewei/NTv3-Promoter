"""Measure fixed-fold-1 iPro-MP runtime, process GPU memory, or forward FLOPs."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.efficiency_common import (  # noqa: E402
    BATCH_SIZE, MAX_LENGTH, NUM_WORKERS, RUNTIME_REPEATS, SPECIES_NAMES,
    WARMUP_BATCHES, atomic_write_json, common_result, configure_device,
    environment_metadata, measure_memory, profile_forward, run_runtime_repeats,
)


OFFICIAL_ROOT = PROJECT_ROOT / "ipro-mp"
DEFAULT_DATA_DIR = PROJECT_ROOT / "external" / "iPro-MP" / "Benchmark Dataset" / "Test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "efficiency" / "ipro_mp"
KNOWN_NONPERSISTENT_BUFFER_KEYS = ("bert.embeddings.position_ids",)


def dependency_compatibility() -> dict[str, Any]:
    usage = {
        "torch": True,
        "transformers": True,
        "peft": False,
        "einops": False,
        "omegaconf": False,
        "accelerate": False,
        "evaluate": False,
    }
    packages: dict[str, Any] = {}
    for name, used in usage.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {
            "version": version,
            "used_by_ipro_mp_benchmark": used,
            "status": "available" if version is not None else "not installed (not on benchmark path)",
        }
    packages["torch"]["runtime_version"] = str(torch.__version__)
    packages["torch"]["cuda_runtime_version"] = torch.version.cuda
    packages["transformers"]["api_check"] = {
        "BertConfig.from_pretrained": callable(getattr(BertConfig, "from_pretrained", None)),
        "BertModel.from_pretrained": callable(getattr(BertModel, "from_pretrained", None)),
        "BertTokenizer.from_pretrained": callable(getattr(BertTokenizer, "from_pretrained", None)),
    }
    if packages["torch"]["version"] is None or packages["transformers"]["version"] is None:
        raise RuntimeError("iPro-MP benchmark requires torch and transformers")
    if not all(packages["transformers"]["api_check"].values()):
        raise RuntimeError("the installed Transformers lacks a required BERT API")
    return {
        "packages": packages,
        "compatibility_policy": (
            "keep promotercls PyTorch/CUDA/Transformers; adapt only the benchmark loader when needed"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "runtime", "memory", "flops"))
    parser.add_argument("--species-id", type=int, default=1, choices=range(1, 24))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--official-root", type=Path, default=OFFICIAL_ROOT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
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
        parser.error("FLOPs uses species 1 fold 1 as the architecture-representative checkpoint")
    if args.mode == "smoke" and args.species_id != 1:
        parser.error("compatibility smoke test is fixed to species 1 fold 1")
    return args


def load_official_module(official_root: Path) -> Any:
    source = official_root / "iPro-MP_predict.py"
    spec = importlib.util.spec_from_file_location("ipro_mp_efficiency_official", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The downloaded official file references BertModel but omits its import.
    module.BertModel = BertModel
    return module


def make_official_model(official: Any, official_root: Path) -> torch.nn.Module:
    dnabert_dir = official_root / "DNABERT-6"
    original = official.BertModel
    weights_present = any(dnabert_dir.glob("*.bin")) or any(dnabert_dir.glob("*.safetensors"))
    if not weights_present:
        config = BertConfig.from_pretrained(str(dnabert_dir))

        class ConfigOnlyBertModel:
            @staticmethod
            def from_pretrained(_: str) -> BertModel:
                return BertModel(config)

        official.BertModel = ConfigOnlyBertModel
    old_cwd = Path.cwd()
    try:
        os.chdir(official_root)
        return official.DNABERTPromoterClassifier()
    finally:
        os.chdir(old_cwd)
        official.BertModel = original


def checkpoint_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"iPro-MP checkpoint must be a dict, got {type(payload)!r}")
    for key in ("trainable_state_dict", "state_dict"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def torch_load_compatibility(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_fold1_model(
    official: Any, official_root: Path, species_id: int, device: torch.device,
) -> tuple[torch.nn.Module, Path, list[str]]:
    checkpoint = official_root / "models" / "07-final" / f"{species_id}_fold_1.pth"
    if not checkpoint.is_file() or checkpoint.name != f"{species_id}_fold_1.pth":
        raise FileNotFoundError(f"required fixed fold-1 checkpoint is missing: {checkpoint}")
    model = make_official_model(official, official_root)
    state_dict = checkpoint_state_dict(
        torch_load_compatibility(checkpoint)
    )
    removed: list[str] = []
    model_keys = set(model.state_dict())
    for key in KNOWN_NONPERSISTENT_BUFFER_KEYS:
        if key in state_dict and key not in model_keys:
            if state_dict[key].is_floating_point():
                raise RuntimeError(f"refusing to remove floating checkpoint value {key}")
            state_dict.pop(key)
            removed.append(key)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    if type(model).__name__ == "OptimizedModule" or hasattr(model, "_orig_mod"):
        raise RuntimeError("torch.compile models are forbidden in this benchmark")
    return model, checkpoint, removed


def parse_fasta_sequences(path: Path) -> list[str]:
    sequences = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    if not sequences:
        raise ValueError(f"no FASTA sequences found in {path}")
    return sequences


def make_loader(
    official: Any, official_root: Path, input_file: Path,
) -> tuple[Any, DataLoader, BertTokenizer]:
    sequences = parse_fasta_sequences(input_file)
    tokenizer = BertTokenizer.from_pretrained(str(official_root / "DNABERT-6"))
    dataset = official.DNADataset(
        sequences=sequences, labels=[0] * len(sequences), tokenizer=tokenizer,
        kmer_size=6, max_len=MAX_LENGTH,
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=False, drop_last=False,
    )
    return dataset, loader, tokenizer


def bf16_autocast_forward(
    model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(input_ids=input_ids, attention_mask=attention_mask)


def warm_up(model: torch.nn.Module, loader: DataLoader, device: torch.device, count: int) -> None:
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            if index >= count:
                break
            logits = bf16_autocast_forward(
                model,
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            torch.softmax(logits, dim=1)[:, 1].cpu().float().numpy()
    torch.cuda.synchronize(device)


def prediction_function(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    expected = len(loader.dataset)

    def predict() -> list[int]:
        probabilities: list[float] = []
        with torch.inference_mode():
            for batch in loader:
                logits = bf16_autocast_forward(
                    model,
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                batch_probabilities = torch.softmax(logits, dim=1)[:, 1]
                probabilities.extend(batch_probabilities.cpu().float().numpy().tolist())
        if len(probabilities) != expected:
            raise RuntimeError(f"iPro-MP predicted {len(probabilities)} samples, expected {expected}")
        return (np.asarray(probabilities) >= 0.5).astype(np.uint8).tolist()

    return predict


def main() -> int:
    args = parse_args()
    device = configure_device(args.device, require_rtx4090=not args.allow_non_4090)
    official_root = args.official_root.resolve()
    input_file = args.data_dir.resolve() / f"{args.species_id}_test.txt"
    official = load_official_module(official_root)
    if list(official.name_list) != SPECIES_NAMES:
        raise ValueError("iPro-MP official species mapping differs from benchmark mapping")
    model, checkpoint, removed_buffers = load_fold1_model(
        official, official_root, args.species_id, device
    )
    dataset, loader, _ = make_loader(official, official_root, input_file)
    pipeline = {
        "source": str(official_root / "iPro-MP_predict.py"),
        "intentional_protocol_change": "fixed fold 1 single model instead of official five-fold ensemble",
        "fold": 1, "model_count": 1, "checkpoint": str(checkpoint),
        "input_file": str(input_file), "kmer_size": 6, "tokenizer": "DNABERT-6 BertTokenizer",
        "online_tokenization": True,
        "prediction": "positive-class softmax probability, threshold >= 0.5, transfer to CPU",
        "precision_mode": "CUDA AMP autocast bfloat16; model tensors retain checkpoint dtypes",
        "compatibility_removed_nonpersistent_buffers": removed_buffers,
        "checkpoint_load_compatibility": (
            "torch.load(weights_only=False) with fallback for PyTorch versions that do not expose "
            "the weights_only keyword; original checkpoint is never rewritten"
        ),
    }
    result = common_result(
        model_key="ipro_mp", mode=args.mode, species_id=args.species_id,
        num_samples=len(dataset), environment=environment_metadata(device), pipeline=pipeline,
    )
    result["dependency_compatibility"] = dependency_compatibility()
    predict = prediction_function(model, loader, device)

    if args.mode == "smoke":
        batch = next(iter(loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with torch.inference_mode():
            logits = bf16_autocast_forward(model, input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
        torch.cuda.synchronize(device)
        expected_shape = (min(BATCH_SIZE, len(dataset)), 2)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(f"smoke output shape is {tuple(logits.shape)}, expected {expected_shape}")
        if not torch.isfinite(logits).all() or not torch.isfinite(probabilities).all():
            raise RuntimeError("smoke output contains NaN or infinity")
        row_sum_error = float((probabilities.sum(dim=1) - 1).abs().max().item())
        if row_sum_error > 1e-3:
            raise RuntimeError(f"invalid softmax probabilities; max row-sum error={row_sum_error}")
        result["smoke_test"] = {
            "status": "passed",
            "scope": "species 1, fixed fold 1, first complete batch",
            "tokenizer_loaded": True,
            "checkpoint_loaded_strictly": True,
            "model_device": str(next(model.parameters()).device),
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "input_ids_shape": list(input_ids.shape),
            "output_shape": list(logits.shape),
            "all_logits_finite": True,
            "all_probabilities_finite": True,
            "softmax_max_row_sum_error": row_sum_error,
        }
        output = args.output_dir / "smoke_test.json"
    elif args.mode == "runtime":
        warm_up(model, loader, device, args.warmup_batches)
        runtime, digest = run_runtime_repeats(predict, device, args.repeats)
        result.update({
            "runtime": runtime,
            "runtime_boundary": (
                "after fixed fold-1 model load, eval, DataLoader creation and BF16 AMP warm-up; "
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
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        if tuple(input_ids.shape) != (1, MAX_LENGTH):
            raise RuntimeError(f"FLOPs input shape is {tuple(input_ids.shape)}, expected (1, 128)")
        profile_text = args.output_dir / "flops_profile.txt"
        result["flops"] = profile_forward(
            model, lambda: bf16_autocast_forward(model, input_ids, attention_mask), profile_text
        )
        result["flops"]["input_ids_shape"] = list(input_ids.shape)
        result["flops"]["fold"] = 1
        result["flops"]["ensemble_multiplier"] = 1
        output = args.output_dir / "flops.json"

    atomic_write_json(output.resolve(), result)
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

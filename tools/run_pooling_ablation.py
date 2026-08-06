#!/usr/bin/env python3
"""Run one pooling ablation serially across species and seeds."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cell.models.ntv3_iPro_mp import POOLING_TYPES
from cell.utils.configs import load_config

CONFIG_RE = re.compile(r"^ntv3_iPro_mp_(?P<name>.+)_(?P<id>\d+)\.yaml$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pooling", required=True, choices=POOLING_TYPES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--config-dir", type=Path, default=PROJECT_ROOT / "configs/models/ntv3_iPro_mp")
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "work_dirs/pooling_ablation")
    parser.add_argument("--result-root", type=Path, default=PROJECT_ROOT / "results/pooling_ablation")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--species-id", type=int, default=None, help="Run one species (sanity checks only).")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16", "fp8"), default="bf16")
    parser.add_argument("--keep-last", action="store_true", help="Also retain the final checkpoint.")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        temporary = Path(f.name)
    temporary.replace(path)


def discover_configs(config_dir: Path) -> list[tuple[int, str, Path]]:
    found = []
    for path in config_dir.glob("ntv3_iPro_mp_*.yaml"):
        match = CONFIG_RE.match(path.name)
        if match:
            raw = OmegaConf.load(path)
            species_name = Path(str(raw.data.train_data.params.data_path)).stem
            found.append((int(match.group("id")), species_name, path.resolve()))
    found.sort()
    ids = [item[0] for item in found]
    if ids != list(range(1, 24)):
        raise RuntimeError(f"Expected exactly species IDs 1..23, found {ids}")
    return found


def effective_config(source: Path, pooling: str, seed: int, max_epochs: int, destination: Path) -> None:
    cfg = load_config(str(source))
    cfg.model.params.pooling_type = pooling
    cfg.seed = seed
    cfg.max_epochs = max_epochs
    cfg._BASE_ = []
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = OmegaConf.load(destination)
        if OmegaConf.to_container(existing, resolve=True) != OmegaConf.to_container(cfg, resolve=True):
            raise RuntimeError(f"Refusing to overwrite a different generated config: {destination}")
    else:
        OmegaConf.save(cfg, destination)


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        raise RuntimeError(f"Training did not create {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or any("AUC" not in row for row in rows):
        raise RuntimeError(f"No complete epoch metrics with AUC in {path}")
    return rows


def retain_checkpoints(run_dir: Path, best_epoch: int, last_epoch: int, keep_last: bool) -> str:
    ckpt_dir = (run_dir / "ckpt").resolve()
    root = run_dir.resolve()
    if ckpt_dir.parent != root:
        raise RuntimeError(f"Unsafe checkpoint path: {ckpt_dir}")
    best = ckpt_dir / f"epoch_{best_epoch}.pth"
    if not best.is_file():
        raise RuntimeError(f"Best checkpoint is missing: {best}")
    retained = ckpt_dir / f"best_epoch_{best_epoch}.pth"
    if not retained.exists():
        payload = torch.load(best, map_location="cpu", weights_only=False)
        if "state_dict" not in payload or int(payload.get("epoch", -1)) != best_epoch:
            raise RuntimeError(f"Malformed best checkpoint: {best}")
        temporary = ckpt_dir / f".best_epoch_{best_epoch}.{uuid.uuid4().hex}.tmp"
        try:
            torch.save({"epoch": best_epoch, "state_dict": payload["state_dict"]}, temporary)
            temporary.replace(retained)
        finally:
            if temporary.exists():
                temporary.unlink()
    verified = torch.load(retained, map_location="cpu", weights_only=False)
    if not retained.is_file() or "state_dict" not in verified or int(verified.get("epoch", -1)) != best_epoch:
        raise RuntimeError(f"Could not verify retained best checkpoint: {retained}")
    keep = {retained.resolve()}
    if keep_last:
        keep.add((ckpt_dir / f"epoch_{last_epoch}.pth").resolve())
    for checkpoint in ckpt_dir.glob("epoch_*.pth"):
        if checkpoint.resolve() not in keep:
            checkpoint.unlink()
    return str(retained)


def main() -> int:
    args = parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain unique integer values")
    if args.max_epochs is not None and args.max_epochs < 1:
        raise ValueError("--max-epochs must be >= 1")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "," in visible_devices:
        raise RuntimeError("Exactly one GPU is required; CUDA_VISIBLE_DEVICES must contain one device")
    disk_probe = args.work_root.resolve()
    if not any(part.startswith("pooling_ablation") for part in disk_probe.parts):
        raise RuntimeError("--work-root must be a dedicated pooling_ablation directory")
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_gb = shutil.disk_usage(disk_probe).free / (1024 ** 3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"Only {free_gb:.2f} GiB free at {disk_probe}; require {args.min_free_gb:.2f} GiB")
    configs = discover_configs(args.config_dir.resolve())
    if args.species_id is not None:
        configs = [item for item in configs if item[0] == args.species_id]
        if not configs:
            raise ValueError(f"Unknown --species-id {args.species_id}")
    total = len(configs) * len(args.seeds)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    ).stdout
    print(f"pooling={args.pooling}\nseeds={args.seeds}\nGPU={visible_devices or '<unset>'}")
    print(f"Python={sys.executable}\nPyTorch={torch.__version__}\nCUDA={torch.version.cuda}\nconfig_count=23")
    print(f"free_disk_gib={free_gb:.2f}")
    print(f"work_root={args.work_root.resolve()}\nresult_root={args.result_root.resolve()}")
    print(f"git_commit={git_commit}\ngit_dirty={bool(git_status.strip())}")
    if git_status:
        print(git_status, end="")
    manifest_path = args.result_root / "manifests" / f"{args.pooling}.json"
    manifest = {
            "pooling": args.pooling,
            "seeds": args.seeds,
            "species_count": len(configs),
            "expected_runs": total,
            "source_configs": [str(item[2]) for item in configs],
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": visible_devices,
            "git_commit": git_commit,
            "git_dirty": bool(git_status.strip()),
            "mixed_precision": args.mixed_precision,
            "max_epochs_override": args.max_epochs,
            "work_root": str(args.work_root.resolve()),
            "result_root": str(args.result_root.resolve()),
        }
    if not args.dry_run:
        if manifest_path.exists():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("pooling", "seeds", "source_configs", "mixed_precision", "max_epochs_override", "work_root"):
                if existing_manifest.get(key) != manifest[key]:
                    raise RuntimeError(f"Existing manifest conflicts on {key}: {manifest_path}")
        else:
            atomic_json(manifest_path, manifest)

    index = 0
    for seed in args.seeds:
        for species_id, species_name, source in configs:
            index += 1
            print(f"[{index}/{total}] pooling={args.pooling} seed={seed} species={species_name}_{species_id}", flush=True)
            run_dir = args.work_root / args.pooling / f"seed_{seed}" / f"{species_id:02d}_{species_name}"
            metadata_path = run_dir / "run_result.json"
            complete_path = run_dir / "COMPLETE"
            if complete_path.is_file() and metadata_path.is_file():
                print("  already complete; skipping")
                continue
            generated = args.result_root / "configs" / args.pooling / f"seed_{seed}" / source.name
            if not args.dry_run:
                effective_config(source, args.pooling, seed, args.max_epochs, generated)
            command = [sys.executable, str(PROJECT_ROOT / "tools/train.py"), str(generated),
                       "--work_dir", str(run_dir), "--seed", str(seed),
                       "--mixed_precision", args.mixed_precision]
            command += ["--max_epochs", str(args.max_epochs)]
            existing = sorted(
                (run_dir / "ckpt").glob("epoch_*.pth"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )
            if existing:
                command += ["--resume_path", str(existing[-1])]
            print("  " + subprocess.list2cmdline(command), flush=True)
            if args.dry_run:
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_dir / "stdout_stderr.log"
            with stdout_path.open("a", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Run failed (exit {completed.returncode}); see {stdout_path}")
            rows = read_metrics(run_dir / "epoch_metrics.jsonl")
            best = max(rows, key=lambda row: row["AUC"])
            checkpoint = retain_checkpoints(run_dir, int(best["epoch"]), int(rows[-1]["epoch"]), args.keep_last)
            normalized = {("MCC" if key == "B_MCC" else key): value for key, value in best.items()}
            payload = {
                "pooling": args.pooling, "seed": seed, "species_id": species_id,
                "species_name": species_name, "config": str(generated), "source_config": str(source),
                "run_dir": str(run_dir), "best_epoch": int(best["epoch"]),
                "best_checkpoint": checkpoint, "effective_config": OmegaConf.to_container(load_config(str(generated))),
                **normalized,
            }
            atomic_json(metadata_path, payload)
            complete_path.write_text("complete\n", encoding="utf-8")
    if not args.dry_run:
        summary_command = [sys.executable, str(PROJECT_ROOT / "tools/summarize_pooling_ablation.py"),
                           "--result-root", str(args.result_root), "--work-root", str(args.work_root)]
        if len(configs) == 23:
            summary_command += ["--pooling", args.pooling]
        subprocess.run(summary_command, cwd=PROJECT_ROOT, check=True)
        print(f"{args.pooling} pooling experiment is complete.")
        print("Inspect its summary before manually starting the next pooling method.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

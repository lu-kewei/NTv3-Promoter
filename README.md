# NTv3-Promoter

NTv3-Promoter fine-tunes the NTv3 8M pretrained model for binary promoter prediction on the 23-species iPro-MP benchmark.

## Requirements

Use Python 3.10 or later and a CUDA-capable PyTorch installation for training.

```bash
conda create -n ntv3-promoter python=3.10
conda activate ntv3-promoter
pip install -r requirements.txt
```

Run commands from the repository root. Training outputs are written under `work_dirs/` and are not intended for version control.

## Data

The study uses the public 23-species Train/Test benchmark from [Jackie-Suv/iPro-MP](https://github.com/Jackie-Suv/iPro-MP). Download it separately, for example:

```bash
git clone https://github.com/Jackie-Suv/iPro-MP.git external/iPro-MP
```

The current NTv3-Promoter dataset loader does not read the upstream FASTA-style `.txt` files directly. It expects one CSV per species at:

```text
data/IPro_MP/
├── train/<species>.csv
└── test/<species>.csv
```

The exact filenames are defined in `configs/models/ntv3_iPro_mp/*.yaml`. Each CSV must contain at least:

```csv
sequence,label_num
ACGT...,1
ACGT...,0
```

`label_num=1` denotes a promoter and `label_num=0` a non-promoter. A raw iPro-MP TXT-to-CSV conversion command is not currently included in this repository; prepare the CSV files before running the supplied configurations.

## Pretrained model

Download the official [InstaDeepAI/NTv3_8M_pre](https://huggingface.co/InstaDeepAI/NTv3_8M_pre) snapshot into the path used by the configurations:

```bash
pip install -U huggingface_hub
hf download InstaDeepAI/NTv3_8M_pre --local-dir models/NTv3_8M_pre
```

Accept the model repository conditions and authenticate with `hf auth login` if Hugging Face requests it. The downloaded model implementation, tokenizer, and weights are third-party artifacts and should not be committed to this repository.

## Training

Train one species by selecting its configuration. For example, species 1 is:

```bash
python tools/train.py \
  configs/models/ntv3_iPro_mp/ntv3_iPro_mp_Acinetobacter_1.yaml \
  --mixed_precision bf16 \
  --seed 42
```

Use `--work_dir` to override the output directory and `--max_epochs` to override the configured epoch count. The directory also contains `ntv3_iPro_mp_all.yaml` for joint training on every CSV in `data/IPro_MP/train/`; its configured in-training test set is one species, so use `tools/test_iPro_all.py` for 23-species evaluation.

## Evaluation and prediction

Evaluate a labeled test CSV with a trained checkpoint:

```bash
python tools/test.py \
  configs/models/ntv3_iPro_mp/ntv3_iPro_mp_Acinetobacter_1.yaml \
  --ckpt_path work_dirs/<run>/ckpt/epoch_<N>.pth \
  --mixed_precision bf16
```

Evaluate a joint-model checkpoint on all 23 configured test CSVs:

```bash
python tools/test_iPro_all.py \
  configs/models/ntv3_iPro_mp/ntv3_iPro_mp_all.yaml \
  --ckpt_path work_dirs/<run>/ckpt/epoch_<N>.pth \
  --mixed_precision bf16
```

These commands compute metrics from labeled CSV files and write logs under `work_dirs/`. The current codebase does not provide a standalone NTv3-Promoter CLI that reads an unlabeled FASTA file and writes per-sequence predictions.

## Efficiency benchmark

The efficiency protocol measures complete-test-set runtime, process GPU memory, and single-sample forward FLOPs. Formal measurements use CUDA, BF16, batch size 64, sequence length 128, and an NVIDIA RTX 4090. Install the additional profiler dependency first:

```bash
pip install -r requirements-efficiency.txt
```

The NTv3 benchmark requires the 23 trained checkpoints under `work_dirs/` and the default checkpoint-selection table at `docs/2026_07_23_17_44_to_2026_07_23_21_19/best_auc_results_all_runs.md`. Run one mode at a time:

```bash
bash scripts/run_efficiency_ntv3.sh runtime 0
bash scripts/run_efficiency_ntv3.sh memory 0
bash scripts/run_efficiency_ntv3.sh flops 0
```

The optional iPro-MP comparison additionally requires its official release files in this local layout:

```text
ipro-mp/
├── iPro-MP_predict.py
├── DNABERT-6/
└── models/07-final/{1..23}_fold_1.pth

external/iPro-MP/Benchmark Dataset/Test/{1..23}_test.txt
```

Then run:

```bash
bash scripts/run_efficiency_ipro_mp.sh runtime 0
bash scripts/run_efficiency_ipro_mp.sh memory 0
bash scripts/run_efficiency_ipro_mp.sh flops 0
python tools/summarize_efficiency.py
```

Generated benchmark JSON, CSV, profiles, and reports are written to `results/efficiency/` and are intentionally ignored by Git.

## Citation

Citation information for NTv3-Promoter will be added with the paper release. Please also cite the original NTv3 and iPro-MP resources when using their model or dataset.

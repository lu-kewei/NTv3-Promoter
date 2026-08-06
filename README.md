# NTv3-based Prokaryotic Promoter Prediction

This repository contains the implementation of a prokaryotic promoter prediction framework based on the pretrained NTv3 model.

The model uses NTv3 to extract sequence representations, followed by self-attention pooling and a binary classification head for promoter and non-promoter prediction.

The project currently supports experiments on 23 prokaryotic species from the iPro-MP benchmark dataset.

> This repository is under active development. 

## Project structure

```text
.
├── cell/                         # Datasets, models, metrics and utilities
├── configs/                      # Training and evaluation configurations
├── data/IPro_MP/                 # Training and test datasets
├── models/NTv3_8M_pre/           # Pretrained NTv3 model
├── tools/
│   ├── train.py                  # Training entry
│   ├── test.py                   # Single-model evaluation
│   └── test_iPro_all.py          # Multi-species evaluation
├── work_dirs/                    # Logs and checkpoints
└── requirements.txt
```

## Installation

Create a Python environment and install the dependencies:

```bash
conda create -n promotercls python=3.10
conda activate promotercls
pip install -r requirements.txt
```

## Data preparation

Place the iPro-MP datasets under:

```text
data/IPro_MP/
├── train/
└── test/
```

Each CSV file should contain at least:

```text
sequence
label_num
```

where `label_num` is the binary class label.

## Pretrained model

Place the pretrained NTv3 model and tokenizer files under:

```text
models/NTv3_8M_pre/
```

The download source and detailed preparation instructions will be provided in a future release.

## Training

Run the following command:

```bash
python tools/train.py \
    configs/models/ntv3_iPro_mp/ntv3_iPro_mp_all.yaml \
    --mixed_precision bf16
```

To specify an output directory:

```bash
python tools/train.py \
    configs/models/ntv3_iPro_mp/ntv3_iPro_mp_all.yaml \
    --work_dir work_dirs/my_experiment \
    --mixed_precision bf16
```

## Evaluation

Evaluate a single checkpoint:

```bash
python tools/test.py \
    configs/models/ntv3_iPro_mp/ntv3_iPro_mp_all.yaml \
    --ckpt_path path/to/checkpoint.pth
```

Run multi-species evaluation:

```bash
python tools/test_iPro_all.py \
    configs/models/ntv3_iPro_mp/ntv3_iPro_mp_all.yaml
```

## Citation

Citation information will be added after publication.

## License

License information will be added before the public release.

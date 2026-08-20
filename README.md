# NTv3-Promoter

This repository contains the implementation of a prokaryotic promoter prediction framework based on the pretrained NTv3 model.

The project currently supports experiments on 23 prokaryotic species from the iPro-MP benchmark dataset.

## Requirements

Use Python 3.10 or later and a CUDA-capable PyTorch installation for training.

```bash
conda create -n ntv3-promoter python=3.10
conda activate ntv3-promoter
pip install -r requirements.txt
```

## Data

The study uses the public 23-species Train/Test benchmark from [Jackie-Suv/iPro-MP](https://github.com/Jackie-Suv/iPro-MP). Download it separately, for example:

```bash
git clone https://github.com/Jackie-Suv/iPro-MP.git external/iPro-MP
```

## Pretrained model

Download the official [InstaDeepAI/NTv3_8M_pre](https://huggingface.co/InstaDeepAI/NTv3_8M_pre) snapshot into the path used by the configurations:

```bash
pip install -U huggingface_hub
hf download InstaDeepAI/NTv3_8M_pre --local-dir models/NTv3_8M_pre
```

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

```b
python tools/test_iPro_all.py \
    configs/models/ntv3_iPro_mp/ntv3_iPro_mp_all.yaml
```


## Efficiency benchmark

Install the additional dependencies:
```bash
pip install -r requirements-efficiency.txt
```
Run the NTv3 efficiency benchmarks:
```bash
bash scripts/run_efficiency_ntv3.sh runtime 0
bash scripts/run_efficiency_ntv3.sh memory 0
bash scripts/run_efficiency_ntv3.sh flops 0
```
For comparison with iPro-MP, prepare the official iPro-MP models and run:
```bash
bash scripts/run_efficiency_ipro_mp.sh runtime 0
bash scripts/run_efficiency_ipro_mp.sh memory 0
bash scripts/run_efficiency_ipro_mp.sh flops 0
```
Summarize the benchmark results with:
```bash
python tools/summarize_efficiency.py
```

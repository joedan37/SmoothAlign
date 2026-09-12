# SmoothAlign

This repository contains the research code for SmoothAlign, a probabilistic
smoothing framework for reducing excessive repulsion between semantically
overlapping non-paired samples in multimodal representation learning.

The repository includes three components:

- `GRAM/`: GRAM volume-based multimodal video-text retrieval, including model,
  data loading, evaluation, and experiment configurations.
- `PCME++/`: image-text matching baselines and SmoothAlign variants, including
  InfoNCE, PCME++, and distributional-prior objectives.
- `IEMOCAP/`: the controlled diagnostic for semantic-overlap correction and
  negative-repulsion weighting.

## Environment

The code was developed with Python 3.9 or newer and PyTorch. GRAM provides a
convenience installation script for its main dependencies:

```bash
cd GRAM
bash preinstall.sh
```

PCME++ has a separate dependency list:

```bash
cd PCME++
python -m pip install -r requirements.txt
```

The exact CUDA, PyTorch, and transformer versions may need to be adjusted to
match the available GPU driver. The code does not download datasets or model
weights automatically.

## GRAM video-text retrieval

Run GRAM from its directory so that the relative configuration paths resolve:

```bash
cd GRAM
python -m torch.distributed.launch \\
  --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_port 9834 \\
  run.py --learning_rate 2e-5 --checkpointing true --first_eval true \\
  --save_best true \\
  --config ./config/gram/finetune_cfg/pretrain-gram.json \\
  --pretrain_dir ./outputs/gram/pretrain_gram \\
  --output_dir ./outputs/gram/pretrain_gram/downstream/pretrain
```

Dataset-specific configurations for MSR-VTT, VATEX, DiDeMo, ActivityNet,
AudioCaps, Flickr, MS-COCO, LSMDC, and YouCook are under
`GRAM/config/gram/finetune_cfg/`. The shell files in
`GRAM/scripts/gram/` contain additional multi-GPU examples.

## PCME++ and SmoothAlign

Train from a YAML configuration using the entry point in `PCME++/`:

```bash
cd PCME++
python train.py ./configs/pcmepp_smooth.yaml
```

The standard baselines and other SmoothAlign settings are available in
`PCME++/configs/`. Configuration values can be overridden with double-underscore
arguments, for example `--optim__lr 0.0001`.

## IEMOCAP controlled diagnostic

The diagnostic requires a locally licensed IEMOCAP copy and an external GRAM
checkpoint. Raw recordings, extracted features, checkpoints, and generated
outputs are intentionally excluded from version control. The expected dataset
layout and required files are documented in `IEMOCAP/README.md`.

Set the resource locations and run the portable scripts:

```bash
export IEMOCAP_DATASET_DIR=/path/to/iemocap
export GRAM_PRETRAIN_DIR=/path/to/gram_pretrained
cd IEMOCAP
bash iemocap_train_portable.sh
bash iemocap_visualize_portable.sh
```

The scripts accept `IEMOCAP_CACHE_DIR`, `IEMOCAP_OUTPUT_DIR`, `PYTHON_BIN`,
`IEMOCAP_CACHE_BATCH_SIZE`, and `IEMOCAP_NUM_WORKERS` overrides. The diagnostic
uses two video frames per utterance and the supplied audio features; see the
script arguments for reproducibility details.

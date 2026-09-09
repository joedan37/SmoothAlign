# IEMOCAP diagnostic

This directory contains the controlled structural-bias diagnostic reported in
the paper.  It is kept separate from `GRAM/` and `PCME++/` while reusing the
shared GRAM model implementation through relative imports.

The raw IEMOCAP recordings, transcripts, extracted audio/video features,
class-anchor cache, checkpoints, logs, and intermediate tensors are omitted.
Prepare a locally licensed dataset and supply an external GRAM checkpoint.
Set `IEMOCAP_DATASET_DIR` and `GRAM_PRETRAIN_DIR`, then run
`iemocap_train_portable.sh`; run `iemocap_visualize_portable.sh` after
training.  Cache and output locations can be overridden with
`IEMOCAP_CACHE_DIR` and `IEMOCAP_OUTPUT_DIR`.  The dataset directory should
contain `train.txt` and `test.txt` (tab-separated sample id, label, and text),
plus `<split>/audio/<id>.pkl` with `audio_feature` and
`<split>/video/<id>.pkl` with `images`.  The checkpoint directory should
contain `log/hps.json` and `ckpt/model_step_249.pt`; shared GRAM encoder
assets are loaded from `../GRAM/weights/encoders/`.  Use Python >=3.9 and set
`PYTHON_BIN` if `python` is not the desired interpreter.

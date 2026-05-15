# Before Start

This document provides a concise workflow to run AuralSAM2 experiments.

## ⚙️ Prepare environment and data

Please complete all setup steps in [installation](./installation.md) first.

## 🚀 Training

Use the unified launcher script:

```bash
cd scripts
./run_avs_train.sh <v1s|v1m|v2> [gpus]
./run_ref_train.sh [gpus]
```
The experiments are implemented by 4 GPUs by default.

## 🔍 Inference (example)

```bash
cd avs.code/v2.code
python inference.py --gpus 1 --batch_size 1 --inference_ckpt /absolute/path/to/checkpoint.pth
```

## 📊 Training Logs (Reproducibility)

Some examples of training details, please see [this wandb link](https://wandb.ai/pyedog1976/AVS-final-report/workspace?nw=nwuserpyedog1976).

In details, after clicking the run (e.g., [v1m-hiera-l](https://wandb.ai/pyedog1976/AVS-final-report/runs/gzp5dmwi/logs?nw=nwuserpyedog1976)), you can checkout:

1) <img src="https://user-images.githubusercontent.com/102338056/167979073-1c1b3144-8a72-4d8d-9084-31d7fdab3e9b.png" width="26" height="22"> overall information (e.g., command line, hardware information and training time).
2) <img src="https://user-images.githubusercontent.com/102338056/167978940-8c1f3d79-d062-4e7b-b56e-30b97d273ae8.png" width="26" height="22"> training curves and validation visualisation.
3) <img src="https://user-images.githubusercontent.com/102338056/167979238-4847430f-aa0b-483d-b735-8a10b43293a1.png" width="26" height="22"> output logs.


## 💾 Checkpoints
We release both checkpoints and training logs in this [Google Drive link](https://drive.google.com/drive/folders/1n0HaCHMn48KaImXvX2mu4qKHUQg4mo9R?usp=sharing).

We also release our checkpoints on Hugging Face: [yyliu01/AuralSAM2](https://huggingface.co/yyliu01/AuralSAM2/tree/main). You can download a weight file directly from the repo **Files** tab, or programmatically with [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub), for example:

```python
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(
    repo_id="yyliu01/AuralSAM2",
    filename="ckpts/auralsam2_avs_v1m.pth",
)
'''
Available Checkpoints:
- `ckpts/auralsam2_avs_v1m.pth`
- `ckpts/auralsam2_avs_v1s.pth`
- `ckpts/auralsam2_avs_v2.pth`
- `ckpts/auralsam2_refavs_best.pth`
'''
```

# Installation

The project is based on Python and PyTorch. We usually run experiments with multi-GPU training.

Tested runtime:
- Python `3.12.3`
- PyTorch `2.8.0+cu128`

## 📥 Clone the Git repo

``` shell
$ https://github.com/yyliu01/AuralSAM2
$ cd AuralSAM2
```

## 🧩 Install dependencies

1) create conda env from yaml
```shell
$ conda env create -f docs/auralsam2.yml
```

2) activate env
```shell
$ conda activate auralsam2
```

3) install PyTorch (recommended: match tested runtime)
```shell
# CUDA 12.8 (tested):
$ pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

4) install python packages (if needed)
```shell
$ pip install -r docs/requirements.txt
```

## 🗂️ Prepare dataset

### AVSBench (`avs.code`)

1) download and prepare AVSBench under repository root.
2) ensure the dataset root path is:
   - `AVSBench/`
   - `AVSBench/avss_index/metadata.csv` (and subset folders `v1s/`, `v1m/`, `v2/`)

### Ref-AVS (`ref-avs.code`)

1) download and prepare the Ref-AVS (REFAVS) dataset under repository root.
2) ensure the dataset root path is:
   - `REFAVS/`
   - `REFAVS/metadata.csv` (splits: `train`, `test_s`, `test_u`, `test_n`)


### Checkpoints (shared)

Prepare under repository root:

- `ckpts/sam_ckpts/sam2_hiera_large.pt`
- `ckpts/vggish-10086976.pth`

## 🏗️ Workspace structure

```shell
AuralSAM2/
├── avs.code/
│   ├── v1s.code/
│   ├── v1m.code/
│   └── v2.code/
├── ref-avs.code/
├── scripts/
│   ├── run_avs_train.sh
│   └── run_ref_train.sh
├── AVSBench/
│   ├── avss_index
│   │   ├── metadata.csv
│   │   ├── metadata_v1m_man.csv
│   │   └── metadata_v2_man.csv
│   ├── v1m
│   │   ├── 01uIJMwnUvA_0
│   │   ├── 0WxgIKuetYI_0
│   │   ... (419 more)
│   ├── v1s
│   │   ├── --FenyW2i_4_5000_10000
│   │   ├── --ZHUMfueO0_5000_10000
│   │   ... (4927 more)
│   └── v2
│       ├── --KCIeTv6PM_14000_24000
│       ├── --iSerV5DbY_68000_78000
│       ... (5995 more)
├── REFAVS/
│   ├── gt_mask
│   │   ├── --KCIeTv6PM_14000_24000
│   │   ├── --iSerV5DbY_68000_78000
│   │   ... (~4000 more)
│   ├── media
│   │   ├── --KCIeTv6PM_14000_24000
│   │   ├── --iSerV5DbY_68000_78000
│   │   ... (~4300 more)
│   └── metadata.csv
├── ckpts/
│   ├── sam_ckpts/
│   │   └── sam2_hiera_large.pt
│   └── vggish-10086976.pth
└── docs/
    ├── installation.md
    ├── before_start.md
    ├── requirements.txt
    └── auralsam2.yml
```

## 📝 Notes

- use `docs/before_start.md` for training and inference commands.
- if wandb is not needed, disable online logging in your config.

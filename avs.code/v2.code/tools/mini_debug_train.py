#!/usr/bin/env python3
"""DDP smoke test: 1 epoch on AVSBench/v2 merge subset (20+20+20 clips).

Build first::

    cd /path/to/v2.code && python3 tools/build_avsbench_v2_merge_subset.py

Then::

    cd /path/to/v2.code && python3 tools/mini_debug_train.py
"""
from __future__ import annotations

import os
import sys

# Avoid MKL + libgomp conflict on some conda stacks before numpy/torch import.
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import numpy  # noqa: F401, E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_REPO)
sys.path.insert(0, _REPO)
_WORKSPACE = os.path.dirname(_REPO)
_MERGE_DATA = os.path.join(_WORKSPACE, "AVSBench", "v2")


def _patch_config() -> None:
    import configs.config as cfg  # noqa: E402

    cfg.C.data_root_path = _MERGE_DATA
    cfg.C.saved_dir = os.path.join("/tmp", "v2_mini_debug_ckpt")
    os.makedirs(cfg.C.saved_dir, exist_ok=True)
    cfg.C.epochs = 1
    cfg.C.batch_size = 1
    cfg.C.num_workers = 0
    cfg.C.wandb_online = False
    cfg.C.gpus = 1


if __name__ == "__main__":
    if not os.path.isdir(_MERGE_DATA):
        raise SystemExit(
            f"missing {_MERGE_DATA} — run: python3 {_REPO}/tools/build_avsbench_v2_merge_subset.py"
        )
    if not os.path.isfile(os.path.join(_MERGE_DATA, "avss_index", "metadata.csv")):
        raise SystemExit(f"missing metadata.csv under {_MERGE_DATA}")

    _patch_config()

    import torch  # noqa: E402
    from easydict import EasyDict  # noqa: E402

    from configs.config import C  # noqa: E402

    hyp = EasyDict(dict(C))
    hyp.gpus = 1
    hyp.batch_size = 1
    hyp.epochs = 1
    hyp.num_workers = 0
    hyp.wandb_online = False
    hyp.data_root_path = _MERGE_DATA
    hyp.saved_dir = os.path.join("/tmp", "v2_mini_debug_ckpt")
    os.makedirs(hyp.saved_dir, exist_ok=True)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "9912")

    from main import main as train_main  # noqa: E402

    torch.multiprocessing.spawn(train_main, nprocs=hyp.gpus, args=(hyp.gpus, hyp))

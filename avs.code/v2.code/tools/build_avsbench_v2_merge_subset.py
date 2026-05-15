#!/usr/bin/env python3
"""
Build merge-debug tree under AVSBench/v2/ (name = v2):

  AVSBench/v2/avss_index/metadata.csv
  AVSBench/v2/v1s/<20 uids>/
  AVSBench/v2/v1m/<20 uids>/
  AVSBench/v2/v2/<20 uids>/   # v2-protocol clips from ~/Downloads/v2.zip

Each modality: 16 train + 4 test rows (20 clips). Full AVSBench is used as source for v1s/v1m copies.
"""
from __future__ import annotations

import csv
import os
import shutil
import zipfile

import pandas as pd

_WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AVSBENCH = os.path.join(_WORKSPACE, "AVSBench")
SUBSET_ROOT = os.path.join(AVSBENCH, "v2")  # data_root for merge run
FULL_META = os.path.join(AVSBENCH, "avss_index", "metadata.csv")
ZIP_PATH = os.path.expanduser("~/Downloads/v2.zip")
PER_LABEL = 20
TRAIN_N, TEST_N = 16, 4


def _pick_rows_from_full_metadata(label: str) -> list[dict]:
    df = pd.read_csv(FULL_META)
    picked: list[dict] = []
    for split, need in ("train", TRAIN_N), ("test", TEST_N):
        got = 0
        sub = df[(df["split"] == split) & (df["label"] == label)]
        for _, row in sub.iterrows():
            uid = str(row["uid"])
            src = os.path.join(AVSBENCH, label, uid)
            if not os.path.isdir(src):
                continue
            picked.append({k: row[k] for k in row.index})
            got += 1
            if got >= need:
                break
        if got < need:
            raise SystemExit(
                f"not enough existing {label}/{split} under {AVSBENCH}: need {need}, got {got}"
            )
    return picked


def _list_v2_uids(z: zipfile.ZipFile) -> list[str]:
    uids: set[str] = set()
    for name in z.namelist():
        if not name.startswith("v2/") or name.endswith("/"):
            continue
        parts = name.split("/")
        if len(parts) >= 3 and parts[1]:
            uids.add(parts[1])
    return sorted(uids)


def _extract_v2_from_zip(uids: list[str]) -> None:
    allowed = set(uids)
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        for info in z.infolist():
            n = info.filename
            if not n.startswith("v2/"):
                continue
            parts = n.split("/")
            if len(parts) < 3 or parts[1] not in allowed:
                continue
            if "/labels_semantic/" in n:
                continue
            if "/frames/" in n and n.endswith(".jpg"):
                pass
            elif "/labels_rgb/" in n and n.endswith(".png"):
                pass
            elif n.endswith("/audio.wav"):
                pass
            else:
                continue
            dest = os.path.join(SUBSET_ROOT, "v2", parts[1], *parts[2:])
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with z.open(info, "r") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)


def _copy_clip(label: str, uid: str) -> None:
    src = os.path.join(AVSBENCH, label, uid)
    dst = os.path.join(SUBSET_ROOT, label, uid)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    if not os.path.isfile(FULL_META):
        raise SystemExit(f"missing full metadata: {FULL_META}")
    if not os.path.isfile(ZIP_PATH):
        raise SystemExit(f"missing v2 zip: {ZIP_PATH}")

    shutil.rmtree(SUBSET_ROOT, ignore_errors=True)
    os.makedirs(os.path.join(SUBSET_ROOT, "avss_index"), exist_ok=True)
    for sub in ("v1s", "v1m", "v2"):
        os.makedirs(os.path.join(SUBSET_ROOT, sub), exist_ok=True)

    all_rows: list[dict] = []

    for label in ("v1s", "v1m"):
        rows = _pick_rows_from_full_metadata(label)
        assert len(rows) == PER_LABEL
        for r in rows:
            _copy_clip(label, str(r["uid"]))
        all_rows.extend(rows)

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        uids_all = _list_v2_uids(z)
    if len(uids_all) < PER_LABEL:
        raise SystemExit(f"v2.zip has only {len(uids_all)} clips")
    v2_uids = uids_all[:PER_LABEL]
    _extract_v2_from_zip(v2_uids)

    for i, uid in enumerate(v2_uids):
        split = "train" if i < TRAIN_N else "test"
        vid = uid.rsplit("_", 2)[0] if uid.count("_") >= 2 else uid
        all_rows.append(
            {
                "vid": vid,
                "uid": uid,
                "s_min": 0,
                "s_sec": 0,
                "a_obj": "v2subset",
                "split": split,
                "label": "v2",
            }
        )

    meta_out = os.path.join(SUBSET_ROOT, "avss_index", "metadata.csv")
    with open(meta_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["vid", "uid", "s_min", "s_sec", "a_obj", "split", "label"])
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    print("subset data_root:", SUBSET_ROOT)
    print("metadata:", meta_out)
    print("v2 uids:", v2_uids)
    print("rows:", len(all_rows))


if __name__ == "__main__":
    main()

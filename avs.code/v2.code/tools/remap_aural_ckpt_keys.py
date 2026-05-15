#!/usr/bin/env python3
"""
Remap legacy checkpoint keys: rename audio_prompter.* to the current AuralFuser layout (aural_fuser.*),
and drop duplicate weights under training_layers / finetuning_layers.

Usage:
  python tools/remap_aural_ckpt_keys.py /path/to/model.pth [--in-place] [--no-backup]

By default writes <stem>_remapped.pth; --in-place overwrites the input (after a .bak backup unless --no-backup).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

# Matches AuralFuser ModuleList names (old train_* indices start at 1; new indices are 0-based).
_REPLACEMENTS: list[tuple[str, str]] = [
    ("train_f_patch_embed1", "patch_embeds.0"),
    ("train_f_patch_embed2", "patch_embeds.1"),
    ("train_f_patch_embed3", "patch_embeds.2"),
    ("train_f_a_block1", "fusion_modules.0"),
    ("train_f_a_block2", "fusion_modules.1"),
    ("train_f_a_block3", "fusion_modules.2"),
    ("train_f_block1", "f_blocks.0"),
    ("train_f_block2", "f_blocks.1"),
    ("train_f_block3", "f_blocks.2"),
    ("train_a_block1", "a_blocks.0"),
    ("train_a_block2", "a_blocks.1"),
    ("train_a_block3", "a_blocks.2"),
    ("train_smooth1", "smooth_convs.0"),
    ("train_smooth2", "smooth_convs.1"),
]


def remap_state_dict(sd: dict) -> dict:
    out: dict = {}
    dropped = 0
    for k, v in sd.items():
        if k.startswith("audio_prompter."):
            if ".training_layers." in k or ".finetuning_layers." in k:
                dropped += 1
                continue
            nk = k.replace("audio_prompter.", "aural_fuser.", 1)
            for old, new in _REPLACEMENTS:
                nk = nk.replace(old, new)
            out[nk] = v
        else:
            out[k] = v
    if dropped:
        print(f"Dropped duplicate keys: {dropped} (training_layers / finetuning_layers)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path, help="Input .pth (full-model state_dict)")
    ap.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output path; default <stem>_remapped.pth",
    )
    ap.add_argument("--in-place", action="store_true", help="Overwrite input file")
    ap.add_argument("--no-backup", action="store_true", help="Skip .bak when using --in-place")
    args = ap.parse_args()

    ckpt_path: Path = args.ckpt.resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"File not found: {ckpt_path}")

    print(f"Loading: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise SystemExit("Expected top-level checkpoint to be a state_dict dict")

    n_old_ap = sum(1 for k in sd if k.startswith("audio_prompter."))
    if n_old_ap == 0:
        print("Warning: no audio_prompter.* keys found; checkpoint may already be remapped.")

    new_sd = remap_state_dict(sd)
    n_af = sum(1 for k in new_sd if k.startswith("aural_fuser."))
    print(f"aural_fuser key count: {n_af}")

    if args.in_place:
        out = ckpt_path
        if not args.no_backup:
            bak = ckpt_path.with_suffix(ckpt_path.suffix + ".bak")
            print(f"Backup -> {bak}")
            shutil.copy2(ckpt_path, bak)
    else:
        out = args.output or ckpt_path.with_name(ckpt_path.stem + "_remapped.pth")

    torch.save(new_sd, out)
    print(f"Saved: {out} ({len(new_sd)} tensor keys)")


if __name__ == "__main__":
    main()

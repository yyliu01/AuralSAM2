#!/usr/bin/env python3
"""
Remap legacy Ref-AVS / AVS checkpoints to the current AuralFuser key layout.

Supports:
  - Full-model ckpt with ``aural_fuser.training_layers.*`` / ``finetuning_layers.*``
    (old PromptAudio ModuleList layout)
  - ``audio_prompter.*`` + ``train_*`` names (older AVS exports)
  - Already-remapped ``patch_embeds.*``, ``f_blocks.*``, etc. (passed through)

Usage (from repo root or ref-avs.code):

  python ref-avs.code/tools/remap_aural_ckpt_keys.py \\
      ckpts/exp/ref-hiera-l/s\\(0.59\\)_u\\(0.68\\).pth \\
      -o ckpts/exp/ref-hiera-l/remapped.pth

Then inference:

  python ref-avs.code/inference.py --gpus 1 \\
      --inference_ckpt ckpts/exp/ref-hiera-l/remapped.pth
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import torch

# Old ``training_layers`` append order in legacy PromptAudio.__init__
_TRAINING_LAYERS_INDEX_MAP: dict[int, str] = {
    0: "patch_embeds.0",
    1: "patch_embeds.1",
    2: "patch_embeds.2",
    3: "f_blocks.0",
    4: "a_blocks.0",
    5: "fusion_modules.0",
    6: "f_blocks.1",
    7: "a_blocks.1",
    8: "fusion_modules.1",
    9: "f_blocks.2",
    10: "a_blocks.2",
    11: "fusion_modules.2",
    12: "smooth_convs.0",
    13: "smooth_convs.1",
    14: "train_proj_v1",
    15: "train_proj_a1",
    16: "text_proj",
}

# Flat ``train_*`` renames (audio_prompter / some aural_fuser exports)
_FLAT_REPLACEMENTS: list[tuple[str, str]] = [
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

_RE_TRAINING_LAYER = re.compile(r"^(?P<prefix>(?:aural_fuser|audio_prompter))\.training_layers\.(\d+)\.(?P<rest>.+)$")
_RE_FINETUNING_LAYER = re.compile(
    r"^(?P<prefix>(?:aural_fuser|audio_prompter))\.finetuning_layers\.0\.(?P<rest>.+)$"
)


def _apply_flat_renames(key: str) -> str:
    for old, new in _FLAT_REPLACEMENTS:
        key = key.replace(old, new)
    return key


def _remap_key(key: str) -> str | None:
    """Return new key, or None to drop the entry."""
    m = _RE_FINETUNING_LAYER.match(key)
    if m:
        prefix = "aural_fuser" if m.group("prefix") == "audio_prompter" else m.group("prefix")
        return f"{prefix}.vgg.{m.group('rest')}"

    m = _RE_TRAINING_LAYER.match(key)
    if m:
        prefix = "aural_fuser" if m.group("prefix") == "audio_prompter" else m.group("prefix")
        idx = int(m.group(2))
        rest = m.group("rest")
        target = _TRAINING_LAYERS_INDEX_MAP.get(idx)
        if target is None:
            return None
        return f"{prefix}.{target}.{rest}"

    if key.startswith("audio_prompter."):
        if ".training_layers." in key or ".finetuning_layers." in key:
            return None
        key = key.replace("audio_prompter.", "aural_fuser.", 1)
        return _apply_flat_renames(key)

    if ".training_layers." in key or ".finetuning_layers." in key:
        return None

    if key.startswith("aural_fuser."):
        return _apply_flat_renames(key)

    return key


def remap_state_dict(sd: dict) -> dict:
    out: dict = {}
    dropped = 0
    remapped = 0
    skip_finetuning = any(k.startswith("aural_fuser.vgg.") for k in sd)
    for k, v in sd.items():
        if skip_finetuning and "finetuning_layers." in k:
            dropped += 1
            continue
        nk = _remap_key(k)
        if nk is None:
            dropped += 1
            continue
        if nk != k:
            remapped += 1
        if nk in out:
            dropped += 1
            continue
        out[nk] = v
    print(f"Remapped keys: {remapped}, dropped: {dropped}")
    return out


def _summarize(sd: dict) -> None:
    prefixes = (
        "v_model.",
        "aural_fuser.patch_embeds",
        "aural_fuser.f_blocks",
        "aural_fuser.vgg",
        "aural_fuser.text_proj",
        "t_model.",
    )
    for p in prefixes:
        n = sum(1 for k in sd if k.startswith(p))
        if n:
            print(f"  {p}*  -> {n} keys")
    legacy = sum(
        1 for k in sd
        if "training_layers" in k or "finetuning_layers" in k or "train_f_patch" in k
    )
    if legacy:
        print(f"  WARNING: {legacy} legacy keys remain")


def main() -> None:
    ap = argparse.ArgumentParser(description="Remap legacy AuralFuser / full-model checkpoint keys")
    ap.add_argument("ckpt", type=Path, help="Input .pth state_dict")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output .pth (default: <stem>_remapped.pth)")
    ap.add_argument("--in-place", action="store_true", help="Overwrite input (creates .bak unless --no-backup)")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument(
        "--aural-fuser-only", action="store_true",
        help="Keep only aural_fuser.* (for aural_fuser-only inference ckpt)",
    )
    args = ap.parse_args()

    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"File not found: {ckpt_path}")

    print(f"Loading: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit("Expected top-level checkpoint to be a state_dict dict")

    n_legacy = sum(
        1 for k in sd
        if "training_layers." in k or "finetuning_layers." in k
    )
    if n_legacy == 0:
        print("Note: no training_layers / finetuning_layers keys; file may already be remapped.")

    new_sd = remap_state_dict(sd)
    if args.aural_fuser_only:
        stripped = {}
        for k, v in new_sd.items():
            if not k.startswith("aural_fuser."):
                continue
            stripped[k[len("aural_fuser."):]] = v
        new_sd = stripped
        print(f"aural-fuser-only (no prefix, for inference.py): {len(new_sd)} keys")

    print("Summary:")
    _summarize(new_sd)

    if args.in_place:
        out = ckpt_path
        if not args.no_backup:
            bak = ckpt_path.with_suffix(ckpt_path.suffix + ".bak")
            print(f"Backup -> {bak}")
            shutil.copy2(ckpt_path, bak)
    else:
        out = args.output or ckpt_path.with_name(ckpt_path.suffix.replace(".pth", "") + "_remapped.pth")

    torch.save(new_sd, out)
    print(f"Saved: {out} ({len(new_sd)} tensor keys)")


if __name__ == "__main__":
    main()

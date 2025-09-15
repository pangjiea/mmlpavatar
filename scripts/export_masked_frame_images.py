#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export a single frame's images from all cameras with masks applied.

Given an AvatarReX-style dataset root, this script:
  - Reads camera names from calibration_full.json
  - Loads the RGB image: <root>/<cam>/<frame:08d>.jpg
  - Loads the mask:     <root>/<cam>/mask/pha/<frame:08d>.jpg
  - Applies the mask, and writes per-camera outputs into --out_dir

Output modes:
  - rgba: Save PNG with alpha from mask (default). Filename: <idx>_<cam>.png
  - rgb_black: Composite on black background (3-channel PNG/JPG)
  - rgb_white: Composite on white background (3-channel PNG/JPG)

Usage:
  python scripts/export_masked_frame_images.py \
      --data_dir /home/hello/data/avatarrex_zzr \
      --frame 1701 \
      --out_dir output/masked_1701 \
      --mode rgba
"""

import argparse
import json
from pathlib import Path

from PIL import Image


def load_cam_names(root: Path):
    calib = root / 'calibration_full.json'
    if not calib.is_file():
        raise FileNotFoundError(f"Not found: {calib}")
    with open(calib, 'r', encoding='utf-8') as f:
        data = json.load(f)
    names = sorted(list(data.keys()))
    return names


def export_frame(root: Path, frame: int, out: Path, mode: str = 'rgba', ext: str = None):
    names = load_cam_names(root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(names)} cameras. Exporting frame {frame} -> {out}")
    frame_name = f"{frame:08d}.jpg"

    if mode == 'rgba':
        ext = 'png'
    elif ext is None:
        ext = 'png' if mode.startswith('rgb_') else 'png'

    ok, miss = 0, 0
    for i, cam in enumerate(names):
        img_path = root / cam / frame_name
        msk_path = root / cam / 'mask' / 'pha' / frame_name
        if not img_path.is_file() or not msk_path.is_file():
            miss += 1
            print(f"Skip {cam}: missing {img_path if not img_path.is_file() else msk_path}")
            continue
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(msk_path).convert('L')  # 0..255

        if mode == 'rgba':
            rgba = img.copy()
            rgba.putalpha(msk)
            save_path = out / f"{i:02d}_{cam}.png"
            rgba.save(save_path)
        elif mode == 'rgb_black':
            bg = Image.new('RGB', img.size, color=(0, 0, 0))
            comp = Image.composite(img, bg, msk)
            save_path = out / f"{i:02d}_{cam}.{ext}"
            comp.save(save_path)
        elif mode == 'rgb_white':
            bg = Image.new('RGB', img.size, color=(255, 255, 255))
            comp = Image.composite(img, bg, msk)
            save_path = out / f"{i:02d}_{cam}.{ext}"
            comp.save(save_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        ok += 1
    print(f"Done. Saved {ok} images, skipped {miss} cameras.")


def main():
    ap = argparse.ArgumentParser(description='Export masked images for a single frame across all cameras')
    ap.add_argument('--data_dir', required=True, help='Dataset root (e.g., /home/hello/data/avatarrex_zzr)')
    ap.add_argument('--frame', type=int, required=True, help='Frame index (e.g., 1701)')
    ap.add_argument('--out_dir', required=True, help='Output directory to save images')
    ap.add_argument('--mode', default='rgba', choices=['rgba', 'rgb_black', 'rgb_white'], help='Output mode')
    ap.add_argument('--ext', default=None, help='Output extension for rgb_* modes (png/jpg). Ignored for rgba.')
    args = ap.parse_args()

    export_frame(Path(args.data_dir), int(args.frame), Path(args.out_dir), mode=args.mode, ext=args.ext)


if __name__ == '__main__':
    main()


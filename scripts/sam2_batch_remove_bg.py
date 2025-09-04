#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two modes in one script:

1) Image batch → masks (existing):
   - Batch segment PNG images and export binary masks to `mask_render`.
   - Uses border negatives + optional auto box from non-white.

2) First-frame annotate → video tracking (new):
   - Annotate foreground on the first frame (clicks and/or ROI box),
     then track and export masks for all frames.
   - Accepts an MP4 path or a directory of frames. If frames are PNGs,
     it converts them to JPEG internally for SAM2.
左键前景、右键背景、b 框选、g 边缘负样本开关、r 重置、Enter/q 继续

Usage (batch images):
  python scripts/sam2_batch_remove_bg.py \
      --input_dir path/to/images \
      [--output_dir path/to/mask_render] \
      [--model facebook/sam2.1-hiera-base-plus] \
      [--device auto]

Usage (video tracking):
  python scripts/sam2_batch_remove_bg.py \
      --video path/to/video.mp4  # or a frames directory \
      [--output_dir path/to/mask_render] \
      [--model facebook/sam2.1-hiera-base-plus] \
      [--device auto]

Dependencies:
  pip install torch pillow numpy opencv-python huggingface_hub
  pip install sam2  # or the official installation method for SAM2

Notes:
  - If CUDA is available, runs with autocast(bfloat16). Falls back to CPU otherwise.
  - If SAM2 import fails, the script will print a helpful message and exit.
  - For MP4 input, SAM2 uses decord under the hood; install decord if needed.
"""

import argparse
import os
import sys
from pathlib import Path
import contextlib
import tempfile
import shutil

import cv2

import json
import numpy as np
from PIL import Image


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def import_sam2_predictor():
    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        return SAM2ImagePredictor
    except Exception as exc:
        eprint("[ERROR] Could not import SAM2ImagePredictor from sam2.")
        eprint("Install dependencies, e.g.:")
        eprint("  pip install huggingface_hub pillow numpy")
        eprint("  pip install sam2  # or follow SAM2 official install instructions")
        eprint(f"Details: {exc}")
        sys.exit(1)


def import_sam2_video_predictor():
    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor  # type: ignore
        return SAM2VideoPredictor
    except Exception as exc:
        eprint("[ERROR] Could not import SAM2VideoPredictor from sam2.")
        eprint("Install dependencies, e.g.:")
        eprint("  pip install huggingface_hub pillow numpy opencv-python")
        eprint("  pip install sam2  # or follow SAM2 official install instructions")
        eprint(f"Details: {exc}")
        sys.exit(1)


def find_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def sample_border_points(width: int, height: int, points_per_side: int = 8) -> np.ndarray:
    """Sample negative (background) points along image borders.

    Returns Nx2 float32 in (x, y) order.
    """
    points = []
    # Ensure at least 2 per side
    n = max(2, int(points_per_side))
    xs = np.linspace(0, width - 1, n, dtype=np.float32)
    ys = np.linspace(0, height - 1, n, dtype=np.float32)

    # Top and bottom rows
    for x in xs:
        points.append((float(x), 0.0))
        points.append((float(x), float(height - 1)))

    # Left and right columns (skip corners to avoid duplicates)
    for y in ys[1:-1]:
        points.append((0.0, float(y)))
        points.append((float(width - 1), float(y)))

    return np.array(points, dtype=np.float32)


def rough_foreground_box(rgb: np.ndarray, white_thresh: int = 250):
    """Compute a rough bounding box of non-white region as (x0,y0,x1,y1), or None.
    White is defined loosely as all channels >= white_thresh.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    non_white = np.any(rgb < white_thresh, axis=2)
    if not np.any(non_white):
        return None, None  # no foreground found
    ys, xs = np.where(non_white)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    # A positive point roughly at the center of the box
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    return (x0, y0, x1, y1), (cx, cy)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_mask(mask_bool: np.ndarray, out_path: Path):
    mask_u8 = (mask_bool.astype(np.uint8)) * 255
    Image.fromarray(mask_u8, mode="L").save(out_path)


def normalize_model_id(model_id: str) -> str:
    """Normalize common typos like underscores to dashes in HF model ids."""
    return model_id.replace("_", "-")


def supports_cv2_highgui() -> bool:
    """Detect if cv2 GUI (imshow/namedWindow) is available."""
    try:
        cv2.namedWindow("_cv2_test_", cv2.WINDOW_NORMAL)
        cv2.imshow("_cv2_test_", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("_cv2_test_")
        return True
    except Exception:
        return False


def parse_points_list(s: str) -> np.ndarray:
    """Parse points from string: "x,y x,y ..." -> Nx2 float32."""
    pts = []
    if not s:
        return np.zeros((0, 2), dtype=np.float32)
    for tok in s.replace(";", " ").split():
        if "," in tok:
            x_str, y_str = tok.split(",")
        else:
            x_str, y_str = tok.split(":") if ":" in tok else tok.split("/")
        pts.append((float(x_str), float(y_str)))
    return np.array(pts, dtype=np.float32)


def parse_box(s: str):
    """Parse box from string: "x0,y0,x1,y1"."""
    if not s:
        return None
    parts = s.replace(" ", "").replace(":", ",").replace("/", ",").split(",")
    if len(parts) != 4:
        raise ValueError("--box expects four numbers: x0,y0,x1,y1")
    x0, y0, x1, y1 = map(float, parts)
    return (x0, y0, x1, y1)


def get_default_prompts_path(video_path: Path, is_dir: bool) -> Path:
    if is_dir:
        return video_path / "sam2_prompts.json"
    else:
        return video_path.with_suffix(".sam2_prompts.json")


def save_prompts_to_file(path: Path, pts: np.ndarray, lbs: np.ndarray, box, frame_w: int, frame_h: int):
    data = {
        "points": pts.tolist(),
        "labels": lbs.astype(int).tolist(),
        "box": list(box) if box is not None else None,
        "image_width": int(frame_w),
        "image_height": int(frame_h),
        "version": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_prompts_from_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pts = np.array(data.get("points", []), dtype=np.float32)
    lbs = np.array(data.get("labels", []), dtype=np.int64)
    box = data.get("box", None)
    if box is not None:
        box = tuple(float(x) for x in box)
    src_w = int(data.get("image_width", 0))
    src_h = int(data.get("image_height", 0))
    return pts, lbs, box, src_w, src_h


def annotate_on_image(first_img_bgr: np.ndarray, auto_border_neg: bool = True, points_per_side: int = 8):
    """Interactive annotation on the first frame using OpenCV.

    - Left click: foreground point
    - Right click: background point
    - Key 'r': reset points
    - Key 'b': select a ROI box via cv2.selectROI
    - Key 'g': toggle adding border negatives (auto)
    - Key 'q' or 'ENTER': finish
    Returns: (points Nx2 float32, labels N int64, box [x0,y0,x1,y1] or None)
    """
    points = []
    labels = []
    box = None
    add_border = auto_border_neg

    disp = first_img_bgr.copy()
    h, w = disp.shape[:2]

    def redraw():
        img = first_img_bgr.copy()
        # draw points
        for (x, y), l in zip(points, labels):
            color = (0, 255, 0) if l == 1 else (0, 0, 255)
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
        # draw box
        if box is not None:
            x0, y0, x1, y1 = map(int, box)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 200, 0), 2)
        # legend
        info = [
            "Left: FG point, Right: BG point",
            "b: select ROI box, r: reset, g: toggle border-neg",
            f"border-neg: {'ON' if add_border else 'OFF'}; ENTER/q: finish",
        ]
        y0 = 20
        for t in info:
            cv2.putText(img, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 22
        return img

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points.append((float(x), float(y)))
            labels.append(0)

    cv2.namedWindow("Annotate First Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotate First Frame", min(1280, w), min(720, h))
    cv2.setMouseCallback("Annotate First Frame", on_mouse)

    while True:
        cv2.imshow("Annotate First Frame", redraw())
        k = cv2.waitKey(20) & 0xFF
        if k in (13, ord('q')):  # ENTER or q
            break
        elif k == ord('r'):
            points.clear(); labels.clear(); box = None
        elif k == ord('b'):
            # ROI returns (x, y, w, h)
            sel = cv2.selectROI("Annotate First Frame", first_img_bgr, fromCenter=False, showCrosshair=True)
            if sel is not None and sel[2] > 0 and sel[3] > 0:
                x0, y0, ww, hh = sel
                box = (float(x0), float(y0), float(x0 + ww - 1), float(y0 + hh - 1))
        elif k == ord('g'):
            add_border = not add_border

    cv2.destroyWindow("Annotate First Frame")

    pts = np.array(points, dtype=np.float32) if points else np.zeros((0, 2), dtype=np.float32)
    lbs = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
    if add_border:
        border_pts = sample_border_points(w, h, points_per_side)
        if border_pts.size:
            pts = np.concatenate([pts, border_pts], axis=0) if pts.size else border_pts
            lbs = np.concatenate([lbs, np.zeros((border_pts.shape[0],), dtype=np.int64)], axis=0)
    return pts, lbs, box


def convert_pngs_to_jpg(src_dir: Path) -> Path:
    """Ensure JPEG frames exist for SAM2 video predictor (expects JPG/JPEG). Returns the dir to use."""
    jpg_dir = src_dir
    # If there is at least one jpg/jpeg, we can use the dir as-is
    has_jpg = any(p.suffix.lower() in {".jpg", ".jpeg"} for p in src_dir.iterdir() if p.is_file())
    has_png = any(p.suffix.lower() == ".png" for p in src_dir.iterdir() if p.is_file())
    if has_jpg or not has_png:
        return jpg_dir

    tmp_dir = src_dir / "_tmp_jpg_for_sam2"
    tmp_dir.mkdir(exist_ok=True)
    pngs = sorted([p for p in src_dir.iterdir() if p.suffix.lower() == ".png"], key=lambda p: int(os.path.splitext(p.name)[0]))
    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            eprint(f"[WARN] failed to read {p} for conversion")
            continue
        out = tmp_dir / (p.stem + ".jpg")
        cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return tmp_dir


def run_video(args):
    from torch import inference_mode, autocast
    import torch

    SAM2VideoPredictor = import_sam2_video_predictor()

    # Resolve input as mp4 or directory
    video_arg = args.video
    if not video_arg:
        eprint("[ERROR] --video is required for video tracking mode")
        sys.exit(2)

    video_path = Path(video_arg)
    if not video_path.exists():
        eprint(f"[ERROR] --video not found: {video_path}")
        sys.exit(2)

    is_dir = video_path.is_dir()
    if is_dir:
        use_dir = convert_pngs_to_jpg(video_path)
        frame_paths_all = [p for p in os.listdir(video_path) if os.path.splitext(p)[-1].lower() in [".png", ".jpg", ".jpeg"]]
        frame_paths_all.sort(key=lambda p: int(os.path.splitext(p)[0]))
        frame_paths_all = [video_path / p for p in frame_paths_all]
        # For saving, keep original filenames; for SAM2, pass the JPEG dir
        sam2_video_input = str(use_dir)
        save_names = [p.name if p.suffix.lower() != ".png" else (p.stem + ".png") for p in frame_paths_all]
        first_frame_to_show = cv2.imread(str(frame_paths_all[0]), cv2.IMREAD_COLOR)
        if first_frame_to_show is None:
            # fallback to reading jpeg counterpart
            first_frame_to_show = cv2.imread(str((use_dir / (Path(frame_paths_all[0]).stem + ".jpg"))), cv2.IMREAD_COLOR)
    else:
        # mp4 file
        sam2_video_input = str(video_path)
        # determine frame count for naming via OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            eprint(f"[ERROR] cannot open video: {video_path}")
            sys.exit(2)
        ret, frame = cap.read()
        if not ret:
            eprint(f"[ERROR] cannot read first frame from: {video_path}")
            sys.exit(2)
        first_frame_to_show = frame
        # If we can get total frame count
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
        cap.release()
        z = 5
        if total is not None:
            z = max(5, len(str(total)))
        save_names = [f"{i:0{z}d}.png" for i in range(0, total or 0)]  # may be refined after reading

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if is_dir:
            output_dir = video_path / "mask_render"
        else:
            output_dir = Path(video_path).parent / "mask_render" / video_path.stem
    ensure_dir(output_dir)

    device = find_device(args.device)
    model_id = normalize_model_id(args.model)
    print(f"Loading video model '{model_id}' on device '{device}' ...")
    predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    # Prepare first-frame prompts: GUI if available and not disabled; else from CLI or auto
    pts = np.zeros((0, 2), dtype=np.float32)
    lbs = np.zeros((0,), dtype=np.int64)
    box = None
    # Prompts save/load path
    cur_h, cur_w = first_frame_to_show.shape[:2]
    prompts_path = Path(args.prompts_path) if args.prompts_path else get_default_prompts_path(video_path if is_dir else Path(video_path), is_dir)
    loaded_from_file = False
    if getattr(args, "load_prompts", False) and prompts_path.exists():
        try:
            l_pts, l_lbs, l_box, src_w, src_h = load_prompts_from_file(prompts_path)
            # Scale if resolution changed
            if src_w > 0 and src_h > 0 and (src_w != cur_w or src_h != cur_h):
                sx = cur_w / float(src_w)
                sy = cur_h / float(src_h)
                if l_pts.size:
                    l_pts = l_pts.copy()
                    l_pts[:, 0] *= sx
                    l_pts[:, 1] *= sy
                if l_box is not None:
                    x0, y0, x1, y1 = l_box
                    l_box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            pts, lbs, box = l_pts, l_lbs, l_box
            loaded_from_file = True
            print(f"Loaded prompts from: {prompts_path}")
        except Exception as exc:
            eprint(f"[WARN] Failed to load prompts from {prompts_path}: {exc}")

    can_gui = supports_cv2_highgui() and not args.no_gui
    if not loaded_from_file and can_gui:
        pts, lbs, box = annotate_on_image(
            first_frame_to_show,
            auto_border_neg=(not args.no_border_neg),
            points_per_side=args.points_per_side,
        )
    elif not loaded_from_file:
        # Try CLI-provided prompts first
        if args.fg:
            fg_pts = parse_points_list(args.fg)
            if fg_pts.size:
                pts = np.concatenate([pts, fg_pts], axis=0)
                lbs = np.concatenate([lbs, np.ones((fg_pts.shape[0],), dtype=np.int64)], axis=0)
        if args.bg:
            bg_pts = parse_points_list(args.bg)
            if bg_pts.size:
                pts = np.concatenate([pts, bg_pts], axis=0)
                lbs = np.concatenate([lbs, np.zeros((bg_pts.shape[0],), dtype=np.int64)], axis=0)
        if args.box:
            box = parse_box(args.box)
        # If still empty, derive autos from first frame
        if pts.size == 0 and box is None:
            rgb = cv2.cvtColor(first_frame_to_show, cv2.COLOR_BGR2RGB)
            auto_box, center = rough_foreground_box(rgb, white_thresh=args.white_threshold)
            box = auto_box
            if center is not None:
                cx, cy = center
                pts = np.array([[float(cx), float(cy)]], dtype=np.float32)
                lbs = np.array([1], dtype=np.int64)
        # Optionally add border negatives in no-GUI mode
        if not args.no_border_neg:
            h, w = first_frame_to_show.shape[:2]
            border_pts = sample_border_points(w, h, args.points_per_side)
            if border_pts.size:
                pts = np.concatenate([pts, border_pts], axis=0) if pts.size else border_pts
                lbs = np.concatenate([lbs, np.zeros((border_pts.shape[0],), dtype=np.int64)], axis=0)

    print(f"First-frame prompts: {len(pts)} pts, box: {'yes' if box is not None else 'no'} (GUI: {'on' if can_gui else 'off'}; loaded: {loaded_from_file})")

    # Auto-save prompts unless disabled
    if not getattr(args, "no_save_prompts", False):
        try:
            save_prompts_to_file(prompts_path, pts, lbs, box, cur_w, cur_h)
            print(f"Saved prompts to: {prompts_path}")
        except Exception as exc:
            eprint(f"[WARN] Failed to save prompts to {prompts_path}: {exc}")

    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_ctx = autocast("cuda", dtype=torch.bfloat16) if is_cuda else contextlib.nullcontext()

    with inference_mode():
        with amp_ctx:
            state = predictor.init_state(sam2_video_input)

            # Option A: stronger first-frame mask by SAM2 image predictor multimask
            use_multimask = getattr(args, "first_frame_multimask", False)
            masks = None
            if use_multimask:
                SAM2ImagePredictor = import_sam2_predictor()
                predictor_img = SAM2ImagePredictor.from_pretrained(model_id, device=device)
                rgb = cv2.cvtColor(first_frame_to_show, cv2.COLOR_BGR2RGB)
                predictor_img.set_image(rgb)
                masks_np, scores_np, _ = predictor_img.predict(
                    point_coords=pts if pts.size else None,
                    point_labels=lbs if lbs.size else None,
                    box=np.array(box, dtype=np.float32) if box is not None else None,
                    multimask_output=True,
                )
                # pick by highest score
                best_idx = int(np.argmax(scores_np)) if scores_np is not None else 0
                best_mask = masks_np[best_idx]
                # ensure boolean 2D
                if best_mask.ndim == 3:
                    best_mask = best_mask[0]
                best_mask = best_mask.astype(bool)
                frame_idx, obj_ids, masks = predictor.add_new_mask(
                    state, frame_idx=0, obj_id=1, mask=best_mask
                )
            else:
                # Option B: points/box
                frame_idx, obj_ids, masks = predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=1,
                    points=pts if pts.size else None,
                    labels=lbs if lbs.size else None,
                    clear_old_points=True,
                    normalize_coords=True,
                    box=box,
                )

            # Save first frame mask
            obj0 = 0
            mask_scores = masks[obj0, 0]  # HxW tensor
            mask_bool = (mask_scores.detach().cpu().numpy() > 0)
            name = (
                save_names[frame_idx]
                if frame_idx < len(save_names)
                else f"{frame_idx:05d}.png"
            )
            out_path = output_dir / name
            save_mask(mask_bool, out_path)
            print(f"[0] -> {out_path}")

            # Propagate to all frames (forward) and save
            for frame_idx, obj_ids, masks in predictor.propagate_in_video(state):
                if frame_idx == 0:
                    continue  # already saved
                mask_scores = masks[obj0, 0]
                mask_bool = (mask_scores.detach().cpu().numpy() > 0)
                name = (
                    save_names[frame_idx]
                    if frame_idx < len(save_names)
                    else f"{frame_idx:05d}.png"
                )
                out_path = output_dir / name
                save_mask(mask_bool, out_path)
                print(f"[{frame_idx}] -> {out_path}")

            # Optional: refine frame 0 at the end using SAM2 image predictor multimask
            if not getattr(args, "no_refine_first", False):
                try:
                    SAM2ImagePredictor = import_sam2_predictor()
                    predictor_img = SAM2ImagePredictor.from_pretrained(model_id, device=device)
                    rgb0 = cv2.cvtColor(first_frame_to_show, cv2.COLOR_BGR2RGB)
                    predictor_img.set_image(rgb0)
                    masks_np, scores_np, _ = predictor_img.predict(
                        point_coords=pts if pts.size else None,
                        point_labels=lbs if lbs.size else None,
                        box=np.array(box, dtype=np.float32) if box is not None else None,
                        multimask_output=True,
                    )
                    best_idx = int(np.argmax(scores_np)) if scores_np is not None else 0
                    best_mask = masks_np[best_idx]
                    if best_mask.ndim == 3:
                        best_mask = best_mask[0]
                    best_mask = best_mask.astype(bool)
                    name0 = save_names[0] if len(save_names) > 0 else f"{0:05d}.png"
                    out0 = output_dir / name0
                    save_mask(best_mask, out0)
                    print(f"[refine-0] -> {out0}")
                except Exception as exc:
                    eprint(f"[WARN] refine first frame failed: {exc}")


def run_images(args):
    from torch import inference_mode, autocast
    import torch

    SAM2ImagePredictor = import_sam2_predictor()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        eprint(f"[ERROR] --input_dir not found: {input_dir}")
        sys.exit(2)

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "mask_render")
    ensure_dir(output_dir)

    device = find_device(args.device)
    model_id = normalize_model_id(args.model)
    print(f"Loading image model '{model_id}' on device '{device}' ...")
    predictor = SAM2ImagePredictor.from_pretrained(model_id, device=device)

    # Collect PNG files
    pngs = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".png"])
    if not pngs:
        eprint(f"[WARN] No PNG files found in {input_dir}")

    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_ctx = autocast("cuda", dtype=torch.bfloat16) if is_cuda else contextlib.nullcontext()

    with inference_mode():
        for idx, img_path in enumerate(pngs, 1):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as exc:
                eprint(f"[WARN] Skip unreadable file: {img_path} ({exc})")
                continue

            rgb = np.array(img)  # H, W, 3 (RGB)
            h, w = rgb.shape[:2]

            # Border negatives
            border_pts = sample_border_points(w, h, args.points_per_side)
            border_labels = np.zeros((border_pts.shape[0],), dtype=np.int64)  # 0 = background

            # Optional rough bbox and a positive point
            box, center = rough_foreground_box(rgb, white_thresh=args.white_threshold)
            point_coords = border_pts
            point_labels = border_labels

            if center is not None:
                cx, cy = center
                pos_pt = np.array([[float(cx), float(cy)]], dtype=np.float32)
                point_coords = np.concatenate([point_coords, pos_pt], axis=0)
                point_labels = np.concatenate([point_labels, np.array([1], dtype=np.int64)], axis=0)  # 1 = foreground

            # Run prediction
            with amp_ctx:
                predictor.set_image(rgb)
                masks = scores = logits = None
                try:
                    masks, scores, logits = predictor.predict(
                        point_coords=point_coords if point_coords.size else None,
                        point_labels=point_labels if point_coords.size else None,
                        box=np.array(box, dtype=np.float32) if box is not None else None,
                        multimask_output=True,
                    )
                except TypeError:
                    # Some versions accept a single prompts dict
                    prompts = {}
                    if point_coords.size:
                        prompts["point_coords"] = point_coords
                        prompts["point_labels"] = point_labels
                    if box is not None:
                        prompts["box"] = np.array(box, dtype=np.float32)
                    masks, scores, logits = predictor.predict(prompts)

            if masks is None:
                eprint(f"[WARN] No masks returned for {img_path}")
                continue

            # Choose best mask by score, fallback to largest area
            if scores is not None and len(scores) == len(masks):
                best_idx = int(np.argmax(scores))
            else:
                areas = [int(m.sum()) for m in masks]
                best_idx = int(np.argmax(areas)) if areas else 0

            best_mask = masks[best_idx]
            # Ensure boolean 2D
            if best_mask.ndim == 3:
                # If it has shape (1, H, W) or (C, H, W)
                best_mask = best_mask[0]
            best_mask = best_mask.astype(bool)

            out_path = output_dir / img_path.name
            save_mask(best_mask, out_path)
            print(f"[{idx}/{len(pngs)}] -> {out_path}")


def build_argparser():
    p = argparse.ArgumentParser(description="SAM2 masking: batch images or video tracking from first-frame prompts")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", help="Folder containing input PNGs for batch image mode")
    group.add_argument("--video", help="MP4 path or frames directory for video tracking mode")
    p.add_argument("--output_dir", default=None, help="Output folder (defaults to <...>/mask_render)")
    p.add_argument("--model", default="facebook/sam2.1-hiera-large", help="Hugging Face model id for SAM2")
    p.add_argument("--device", default="auto", help="Device: auto|cuda|cpu")
    p.add_argument("--points_per_side", type=int, default=8, help="Negative points per border side (and border-neg in first-frame UI)")
    p.add_argument("--white_threshold", type=int, default=250, help="RGB channel threshold for white background (batch image + video auto mode)")
    # Video mode extras
    p.add_argument("--no_gui", action="store_true", help="Disable OpenCV GUI; use CLI prompts or auto prompts")
    p.add_argument("--no_border_neg", action="store_true", help="Disable auto border negative points on first frame")
    p.add_argument("--first_frame_multimask", action="store_true", help="Use SAM2 image predictor to pick best of multiple masks on first frame")
    p.add_argument("--no_refine_first", action="store_true", help="Do not recompute and overwrite frame-0 mask at the end")
    p.add_argument("--load_prompts", action="store_true", help="Load saved first-frame prompts from JSON before annotation")
    p.add_argument("--no_save_prompts", action="store_true", help="Disable auto-saving first-frame prompts to JSON")
    p.add_argument("--prompts_path", default=None, help="Custom path to save/load prompts JSON (defaults beside video or in frames dir)")
    p.add_argument("--box", default=None, help="First-frame box: 'x0,y0,x1,y1' (video mode)")
    p.add_argument("--fg", default=None, help="First-frame foreground points: 'x,y x,y ...' (video mode)")
    p.add_argument("--bg", default=None, help="First-frame background points: 'x,y x,y ...' (video mode)")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.video:
        run_video(args)
    else:
        run_images(args)

#!/usr/bin/env python3
"""
PanoVGGT — Inference Script
Input  : panoramic images + optional masks (black = invalid)
Output : depth maps, camera poses (rotation matrices), per-frame PLY, merged PLY
"""

import os
import sys
import argparse
import glob
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from panovggt.models.panovggt_model import PanoVGGTModel
from panovggt.utils.basic import load_images_as_tensor

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# =========================================================================
#  1.  Model Loading
# =========================================================================

def load_model(config_path: str, checkpoint_path: str, device: str) -> PanoVGGTModel:
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    mc = cfg.model
    model = PanoVGGTModel(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        enable_camera=mc.enable_camera,
        enable_depth=mc.enable_depth,
        enable_point=mc.enable_point,
        aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if key in ckpt:
            ckpt = ckpt[key]
            break
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load] missing keys  : {missing[:5]}{'…' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[load] unexpected keys: {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
    print("✅ PanoVGGT model loaded successfully.")
    return model


# =========================================================================
#  2.  I/O Helpers
# =========================================================================

def collect_images(image_dir: str) -> List[str]:
    """Return sorted image paths from a directory."""
    paths = sorted(
        p for p in glob.glob(os.path.join(image_dir, "*"))
        if os.path.splitext(p)[1].lower() in _IMG_EXTS
    )
    if not paths:
        raise ValueError(f"No images found in: {image_dir}")
    return paths


def collect_masks(mask_dir: Optional[str], image_paths: List[str]) -> Optional[List[Optional[str]]]:
    """
    Match mask files to image files by stem name.
    Returns None if mask_dir is None, else a list (same length as image_paths)
    where each element is either a path string or None.
    """
    if mask_dir is None:
        return None

    mask_map = {}
    for p in glob.glob(os.path.join(mask_dir, "*")):
        if os.path.splitext(p)[1].lower() in _IMG_EXTS:
            mask_map[Path(p).stem] = p

    result = []
    for img_path in image_paths:
        stem = Path(img_path).stem
        result.append(mask_map.get(stem, None))
    return result


def load_mask(mask_path: Optional[str], H: int, W: int) -> Optional[np.ndarray]:
    """
    Load a mask and return a boolean array (H, W) where True = valid pixel.
    Black pixels (all channels == 0) are treated as invalid.
    Returns None if mask_path is None.
    """
    if mask_path is None:
        return None
    mask_bgr = cv2.imread(mask_path)
    if mask_bgr is None:
        print(f"  [warn] Could not read mask: {mask_path}")
        return None
    mask_bgr = cv2.resize(mask_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
    # valid where at least one channel > 0
    valid = np.any(mask_bgr > 0, axis=-1)   # (H, W) bool
    return valid


# =========================================================================
#  3.  Depth Visualisation
# =========================================================================

def depth_to_colormap(
    depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    colormap: int = cv2.COLORMAP_TURBO,
    use_log: bool = True,
) -> np.ndarray:
    """
    Convert a (H, W) float depth array to a uint8 BGR colour image.
    Invalid pixels (mask == False) are rendered black.
    """
    d = depth.copy().astype(np.float32)

    if valid_mask is not None:
        d[~valid_mask] = np.nan

    if use_log:
        d = np.log1p(d)

    d_min = np.nanmin(d)
    d_max = np.nanmax(d)
    if d_max - d_min < 1e-6:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    else:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    # nan → 0 before colormap
    d_norm = np.nan_to_num(d_norm, nan=0).astype(np.uint8)
    colored = cv2.applyColorMap(d_norm, colormap)

    if valid_mask is not None:
        colored[~valid_mask] = 0

    return colored


# =========================================================================
#  4.  Point Cloud Helpers
# =========================================================================

def points_and_colors_from_frame(
    points_hw3: np.ndarray,
    image_hw3: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten (H, W, 3) points and image arrays, applying optional mask.
    Returns (N, 3) xyz and (N, 3) rgb (uint8).
    """
    H, W, _ = points_hw3.shape
    mask = valid_mask if valid_mask is not None else np.ones((H, W), dtype=bool)

    # additional validity: discard points with zero depth / NaN
    depth = np.linalg.norm(points_hw3, axis=-1)
    mask = mask & (depth > 0) & np.isfinite(depth)

    xyz = points_hw3[mask]                    # (N, 3)
    rgb = (image_hw3[mask] * 255).astype(np.uint8) if image_hw3.max() <= 1.0 \
          else image_hw3[mask].astype(np.uint8)

    return xyz, rgb


def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a coloured point cloud to a PLY file (ASCII)."""
    assert xyz.shape[0] == rgb.shape[0], "xyz / rgb length mismatch"
    N = xyz.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    print(f"  [ply] saved {N:,} points → {path}")


# =========================================================================
#  5.  Inference
# =========================================================================

def run_inference(
    model: PanoVGGTModel,
    image_dir: str,
    device: str,
) -> dict:
    """Run PanoVGGT on all images in image_dir. Returns raw prediction dict."""
    imgs = load_images_as_tensor(image_dir, interval=1).to(device)
    print(f"[infer] Input tensor shape: {imgs.shape}")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(imgs[None])   # add batch dim → (1, S, C, H, W)

    # Convert bfloat16 → float32, move to CPU
    out = {}
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            out[k] = (v.float() if v.dtype == torch.bfloat16 else v).cpu()
        else:
            out[k] = v

    # squeeze batch dim
    for k in list(out.keys()):
        if isinstance(out[k], torch.Tensor) and out[k].shape[0] == 1:
            out[k] = out[k].squeeze(0)

    return out


# =========================================================================
#  6.  Main Pipeline
# =========================================================================

def main(args):
    # ── device ────────────────────────────────────────────────────────────
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.")
        device = "cpu"

    # ── output dirs ───────────────────────────────────────────────────────
    out_root      = args.output_dir
    depth_dir     = os.path.join(out_root, "depth")
    pose_dir      = os.path.join(out_root, "poses")
    per_frame_dir = os.path.join(out_root, "pointclouds", "per_frame")
    merged_dir    = os.path.join(out_root, "pointclouds")
    for d in (depth_dir, pose_dir, per_frame_dir, merged_dir):
        os.makedirs(d, exist_ok=True)

    # ── collect inputs ────────────────────────────────────────────────────
    image_paths = collect_images(args.image_dir)
    mask_paths  = collect_masks(args.mask_dir, image_paths)
    S           = len(image_paths)
    print(f"[pipeline] {S} image(s) found.")
    if mask_paths:
        n_masks = sum(1 for m in mask_paths if m is not None)
        print(f"[pipeline] {n_masks}/{S} mask(s) matched.")

    # ── model ─────────────────────────────────────────────────────────────
    model = load_model(args.config, args.checkpoint, device)
    model.eval().to(device)
    print(f"[pipeline] Model ready on {device}.")

    # ── inference ─────────────────────────────────────────────────────────
    # load_images_as_tensor expects a directory; we use args.image_dir directly
    preds = run_inference(model, args.image_dir, device)

    # ── decode outputs ────────────────────────────────────────────────────
    # preds shapes (after squeeze):
    #   depth         : (S, H, W) or (S, H, W, 1)
    #   local_points  : (S, H, W, 3)
    #   world_points  : (S, H, W, 3)   ← preferred for merged cloud
    #   camera_poses  : (S, 4, 4)
    #   images        : (S, C, H, W)   raw input tensor

    # depth
    depth_np = None
    if "depth" in preds and preds["depth"] is not None:
        depth_np = preds["depth"].numpy()               # (S, H, W) or (S, H, W, 1)
        if depth_np.ndim == 4:
            depth_np = depth_np[..., 0]                 # → (S, H, W)

    # points in world frame (used for merged cloud)
    world_pts_np = None
    if "world_points" in preds and preds["world_points"] is not None:
        world_pts_np = preds["world_points"].numpy()    # (S, H, W, 3)
    elif "points" in preds and preds["points"] is not None:
        world_pts_np = preds["points"].numpy()

    # points in local/camera frame (used for per-frame cloud)
    local_pts_np = None
    if "local_points" in preds and preds["local_points"] is not None:
        local_pts_np = preds["local_points"].numpy()    # (S, H, W, 3)
    elif world_pts_np is not None:
        local_pts_np = world_pts_np                     # fall back

    # camera poses  (S, 4, 4)
    poses_np = None
    if "camera_poses" in preds and preds["camera_poses"] is not None:
        poses_np = preds["camera_poses"].numpy()

    # input images normalised to [0, 1], shape (S, C, H, W) or (S, H, W, C)
    images_raw = None
    if "images" in preds and preds["images"] is not None:
        images_raw = preds["images"]
        if isinstance(images_raw, torch.Tensor):
            images_raw = images_raw.numpy()
        if images_raw.ndim == 4 and images_raw.shape[1] in (1, 3):
            # (S, C, H, W) → (S, H, W, C)
            images_raw = images_raw.transpose(0, 2, 3, 1)
        if images_raw.max() > 1.0:
            images_raw = images_raw / 255.0

    # derive H, W from depth or points
    if depth_np is not None:
        _, H, W = depth_np.shape
    elif local_pts_np is not None:
        _, H, W, _ = local_pts_np.shape
    else:
        # fall back: read first image
        tmp = cv2.imread(image_paths[0])
        H, W = tmp.shape[:2]

    print(f"[pipeline] Frame size: {H} × {W}")

    # ── per-frame output ──────────────────────────────────────────────────
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []

    for i, img_path in enumerate(image_paths):
        stem = Path(img_path).stem
        print(f"\n[frame {i:04d}] {stem}")

        # load original image for colour (RGB, float32 [0,1])
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = cv2.resize(img_rgb, (W, H))

        # load mask
        mask_valid = None
        if mask_paths is not None and mask_paths[i] is not None:
            mask_valid = load_mask(mask_paths[i], H, W)
            print(f"  mask: {mask_paths[i]}  valid={mask_valid.sum():,}/{H*W:,} px")
        else:
            print("  mask: (none)")

        # ── depth map ─────────────────────────────────────────────────────
        if depth_np is not None:
            d = depth_np[i]                                     # (H, W)
            colored = depth_to_colormap(
                d, valid_mask=mask_valid,
                use_log=args.log_depth,
                colormap=cv2.COLORMAP_TURBO,
            )
            out_depth_path = os.path.join(depth_dir, f"{stem}_depth.png")
            cv2.imwrite(out_depth_path, colored)

            # also save raw depth as 16-bit PNG (millimetres, clamped to 65535)
            d_mm = (d * 1000.0).astype(np.float32)
            if mask_valid is not None:
                d_mm[~mask_valid] = 0.0
            d_u16 = np.clip(d_mm, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(depth_dir, f"{stem}_depth_raw.png"), d_u16)
            print(f"  depth: saved → {out_depth_path}")

        # ── camera pose ───────────────────────────────────────────────────
        if poses_np is not None:
            pose44 = poses_np[i]                                # (4, 4)
            R      = pose44[:3, :3]                             # (3, 3)
            t      = pose44[:3,  3]                             # (3,)

            np.savetxt(
                os.path.join(pose_dir, f"{stem}_R.txt"), R,
                fmt="%.8f", header=f"Rotation matrix for frame {i}: {stem}")
            np.savetxt(
                os.path.join(pose_dir, f"{stem}_t.txt"), t,
                fmt="%.8f", header=f"Translation for frame {i}: {stem}")
            np.save(os.path.join(pose_dir, f"{stem}_pose.npy"), pose44)
            print(f"  pose: R saved, t={t}")

        # ── per-frame point cloud (local frame) ───────────────────────────
        if local_pts_np is not None:
            pts_hw3 = local_pts_np[i]                           # (H, W, 3)
            xyz, rgb = points_and_colors_from_frame(
                pts_hw3, img_rgb, valid_mask=mask_valid)
            ply_path = os.path.join(per_frame_dir, f"{stem}.ply")
            save_ply(ply_path, xyz, rgb)

        # ── accumulate for merged cloud (world frame) ─────────────────────
        if world_pts_np is not None:
            pts_hw3 = world_pts_np[i]
            xyz_w, rgb_w = points_and_colors_from_frame(
                pts_hw3, img_rgb, valid_mask=mask_valid)
            all_xyz.append(xyz_w)
            all_rgb.append(rgb_w)

    # ── merged point cloud ────────────────────────────────────────────────
    if all_xyz:
        merged_xyz = np.concatenate(all_xyz, axis=0)
        merged_rgb = np.concatenate(all_rgb, axis=0)
        save_ply(os.path.join(merged_dir, "merged.ply"), merged_xyz, merged_rgb)

    # ── save all poses together ───────────────────────────────────────────
    if poses_np is not None:
        np.save(os.path.join(pose_dir, "all_poses.npy"), poses_np)   # (S, 4, 4)
        # also save rotation matrices only: (S, 3, 3)
        np.save(os.path.join(pose_dir, "all_rotations.npy"), poses_np[:, :3, :3])
        print(f"\n[pipeline] All poses saved → {pose_dir}/all_poses.npy")

    print(f"\n✅ Done.  Results written to: {out_root}")


# =========================================================================
#  7.  Entry Point
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PanoVGGT Inference — depth, poses, point clouds")

    # required
    p.add_argument("--config",      type=str, required=True,
                   help="Path to model YAML config file.")
    p.add_argument("--checkpoint",  type=str, required=True,
                   help="Path to model checkpoint (.pth / .pt).")
    p.add_argument("--image_dir",   type=str, required=True,
                   help="Directory containing panoramic input images.")

    # optional
    p.add_argument("--mask_dir",    type=str, default=None,
                   help="Directory containing masks (same stem as images). "
                        "Black pixels = invalid. Omit to skip masking.")
    p.add_argument("--output_dir",  type=str, default="panovggt_output",
                   help="Root output directory. (default: panovggt_output)")
    p.add_argument("--device",      type=str, default="cuda",
                   choices=["cuda", "cpu"],
                   help="Compute device. (default: cuda)")
    p.add_argument("--log_depth",   action="store_true", default=True,
                   help="Use logarithmic scale for depth visualisation. (default: True)")
    p.add_argument("--no_log_depth", dest="log_depth", action="store_false",
                   help="Disable logarithmic depth scale.")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
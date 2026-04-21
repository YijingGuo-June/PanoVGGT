#!/usr/bin/env python3
"""
PanoVGGT — Inference Script
Input  : panoramic images + optional masks (black = invalid)
Output : depth maps, camera poses (rotation matrices), per-frame PLY, merged PLY

python inference.py \
    --config  training/config/default.yaml \
    --checkpoint training/checkpoints/model.pt \
    --image_dir  data/image \
    --mask_dir   data/image_mask \
    --output_dir results

"""

import os
import argparse
import contextlib
import glob
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from panovggt.models.panovggt_model import PanoVGGTModel

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

# 固定输入分辨率 (H=518, W=1036，宽高比 2:1，均能被 patch_size=14 整除)
_INPUT_H = 518
_INPUT_W = 1036


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


def collect_masks(
    mask_dir: Optional[str],
    image_paths: List[str],
) -> Optional[List[Optional[str]]]:
    if mask_dir is None:
        return None

    mask_map: dict = {}
    for p in glob.glob(os.path.join(mask_dir, "*")):
        if os.path.splitext(p)[1].lower() in _IMG_EXTS:
            stem = Path(p).stem
            # 去掉常见的 mask 后缀，统一用图片 stem 作为 key
            for suffix in ("_mask", "-mask", "_Mask", "-Mask"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            mask_map[stem] = p

    # Debug
    print(f"[mask] image stems : {[Path(p).stem for p in image_paths]}")
    print(f"[mask] mask  stems (normalised) : {sorted(mask_map.keys())}")

    result = []
    for img_path in image_paths:
        stem = Path(img_path).stem
        matched = mask_map.get(stem, None)
        if matched is None:
            print(f"  [mask] WARNING: no mask found for '{stem}'")
        result.append(matched)
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
    valid = np.any(mask_bgr > 0, axis=-1)  # (H, W) bool
    return valid


# =========================================================================
#  3.  Image Loading  (fixed 518 × 1036)
# =========================================================================

def load_images_fixed(image_paths: List[str]) -> torch.Tensor:
    """
    Load images, resize each to (_INPUT_H, _INPUT_W), normalise to [0,1].
    Returns float32 tensor of shape (S, 3, _INPUT_H, _INPUT_W).
    """
    frames = []
    for p in image_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            raise IOError(f"Cannot read image: {p}")
        bgr = cv2.resize(bgr, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # (H, W, 3) → (3, H, W)
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    return torch.stack(frames, dim=0)  # (S, 3, H, W)


# =========================================================================
#  4.  Depth Visualisation
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

    d_norm = np.nan_to_num(d_norm, nan=0).astype(np.uint8)
    colored = cv2.applyColorMap(d_norm, colormap)

    if valid_mask is not None:
        colored[~valid_mask] = 0

    return colored


# =========================================================================
#  5.  Point Cloud Helpers
# =========================================================================

def points_and_colors_from_frame(
    points_hw3: np.ndarray,
    image_hw3: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten (H, W, 3) points and image arrays, applying optional mask.
    Returns (N, 3) xyz and (N, 3) rgb uint8.
    """
    H, W, _ = points_hw3.shape
    mask = valid_mask if valid_mask is not None else np.ones((H, W), dtype=bool)

    # discard zero-depth / NaN points
    depth = np.linalg.norm(points_hw3, axis=-1)
    mask = mask & (depth > 0) & np.isfinite(depth)

    xyz = points_hw3[mask]  # (N, 3)
    if image_hw3.max() <= 1.0:
        rgb = (image_hw3[mask] * 255).astype(np.uint8)
    else:
        rgb = image_hw3[mask].astype(np.uint8)

    return xyz, rgb


def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a coloured point cloud to a PLY file (binary little-endian for speed)."""
    assert xyz.shape[0] == rgb.shape[0], "xyz / rgb length mismatch"
    N = xyz.shape[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    xyz = xyz.astype(np.float32)
    rgb = rgb.astype(np.uint8)

    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        # interleave xyz (float32 × 3) and rgb (uint8 × 3) per vertex
        data = np.empty(N, dtype=[
            ("x", np.float32), ("y", np.float32), ("z", np.float32),
            ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
        ])
        data["x"] = xyz[:, 0]
        data["y"] = xyz[:, 1]
        data["z"] = xyz[:, 2]
        data["r"] = rgb[:, 0]
        data["g"] = rgb[:, 1]
        data["b"] = rgb[:, 2]
        f.write(data.tobytes())

    print(f"  [ply] saved {N:,} points → {path}")


# =========================================================================
#  6.  Inference
# =========================================================================

def run_inference(
    model: PanoVGGTModel,
    image_paths: List[str],
    device: str,
) -> dict:
    """
    Run PanoVGGT on the given image paths.
    Images are resized to (_INPUT_H, _INPUT_W) = (518, 1036) before inference.
    Returns a dict of numpy arrays (batch dim already squeezed).
    """
    imgs = load_images_fixed(image_paths).to(device)
    print(f"[infer] Input tensor shape: {imgs.shape}  "
          f"(S={imgs.shape[0]}, C=3, H={_INPUT_H}, W={_INPUT_W})")

    # autocast only on CUDA
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    with torch.no_grad(), amp_ctx:
        # imgs: (S, 3, H, W) → add batch dim → (1, S, 3, H, W)
        preds = model(imgs.unsqueeze(0))

    # Convert to float32 CPU numpy, squeeze batch dim
    out: dict = {}
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            v_f = v.float() if v.dtype == torch.bfloat16 else v
            v_cpu = v_f.cpu()
            # squeeze leading batch dim if present
            if v_cpu.dim() >= 1 and v_cpu.shape[0] == 1:
                v_cpu = v_cpu.squeeze(0)
            out[k] = v_cpu.numpy()
        else:
            out[k] = v

    return out


# =========================================================================
#  7.  Main Pipeline
# =========================================================================

def main(args: argparse.Namespace) -> None:
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
    S = len(image_paths)
    print(f"[pipeline] {S} image(s) found.")
    if mask_paths is not None:
        n_masks = sum(1 for m in mask_paths if m is not None)
        print(f"[pipeline] {n_masks}/{S} mask(s) matched.")

    # ── model ─────────────────────────────────────────────────────────────
    model = load_model(args.config, args.checkpoint, device)
    model.eval().to(device)
    print(f"[pipeline] Model ready on {device}.")
    print(f"[pipeline] Fixed input resolution: H={_INPUT_H}, W={_INPUT_W}")

    # ── inference ─────────────────────────────────────────────────────────
    preds = run_inference(model, image_paths, device)

    # ── unpack predictions ────────────────────────────────────────────────
    # After squeeze, expected shapes:
    #   depth        : (S, H, W) or (S, H, W, 1)
    #   local_points : (S, H, W, 3)   — camera / local frame
    #   world_points : (S, H, W, 3)   — world frame  (preferred for merged cloud)
    #   camera_poses : (S, 4, 4)
    #   images       : (S, C, H, W) or (S, H, W, C)  — model's view of the input

    # depth  (S, H, W)
    depth_np: Optional[np.ndarray] = None
    if "depth" in preds and preds["depth"] is not None:
        depth_np = preds["depth"]
        if depth_np.ndim == 4:          # (S, H, W, 1) → (S, H, W)
            depth_np = depth_np[..., 0]

    # world-frame points  (S, H, W, 3)
    world_pts_np: Optional[np.ndarray] = None
    if "world_points" in preds and preds["world_points"] is not None:
        world_pts_np = preds["world_points"]
    elif "points" in preds and preds["points"] is not None:
        world_pts_np = preds["points"]

    # local-frame points  (S, H, W, 3)
    local_pts_np: Optional[np.ndarray] = None
    if "local_points" in preds and preds["local_points"] is not None:
        local_pts_np = preds["local_points"]
    elif world_pts_np is not None:
        local_pts_np = world_pts_np     # fall back

    # camera poses  (S, 4, 4)
    poses_np: Optional[np.ndarray] = None
    if "camera_poses" in preds and preds["camera_poses"] is not None:
        poses_np = preds["camera_poses"]

    # H, W at inference resolution
    H, W = _INPUT_H, _INPUT_W

    print(f"[pipeline] Inference frame size: {H} × {W}")

    # ── per-frame processing ──────────────────────────────────────────────
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []

    for i, img_path in enumerate(image_paths):
        stem = Path(img_path).stem
        print(f"\n[frame {i:04d}] {stem}")

        # Load original image, resize to inference resolution for colour lookup
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  [warn] Cannot read image: {img_path}. Skipping.")
            continue
        img_bgr_resized = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load and resize mask to inference resolution
        mask_valid: Optional[np.ndarray] = None
        if mask_paths is not None and mask_paths[i] is not None:
            mask_valid = load_mask(mask_paths[i], H, W)
            if mask_valid is not None:
                print(f"  mask: {mask_paths[i]}  "
                      f"valid={mask_valid.sum():,}/{H * W:,} px "
                      f"({100.0 * mask_valid.mean():.1f} %)")
        else:
            print("  mask: (none — all pixels valid)")

        # ── depth map ─────────────────────────────────────────────────────
        if depth_np is not None:
            d = depth_np[i]                                 # (H, W)

            # coloured visualisation
            colored = depth_to_colormap(
                d,
                valid_mask=mask_valid,
                use_log=args.log_depth,
                colormap=cv2.COLORMAP_TURBO,
            )
            out_depth_path = os.path.join(depth_dir, f"{stem}_depth.png")
            cv2.imwrite(out_depth_path, colored)

            # raw 16-bit depth (millimetres, masked invalid → 0)
            d_mm = (d * 1000.0).astype(np.float32)
            if mask_valid is not None:
                d_mm[~mask_valid] = 0.0
            d_u16 = np.clip(d_mm, 0, 65535).astype(np.uint16)
            cv2.imwrite(
                os.path.join(depth_dir, f"{stem}_depth_raw.png"), d_u16
            )
            print(f"  depth: saved → {out_depth_path}")

        # ── camera pose ───────────────────────────────────────────────────
        if poses_np is not None:
            pose44 = poses_np[i]                            # (4, 4)
            R = pose44[:3, :3]                              # (3, 3)
            t = pose44[:3, 3]                               # (3,)

            np.savetxt(
                os.path.join(pose_dir, f"{stem}_R.txt"), R,
                fmt="%.8f",
                header=f"Rotation matrix for frame {i}: {stem}",
            )
            np.savetxt(
                os.path.join(pose_dir, f"{stem}_t.txt"), t[np.newaxis],
                fmt="%.8f",
                header=f"Translation for frame {i}: {stem}",
            )
            np.save(os.path.join(pose_dir, f"{stem}_pose.npy"), pose44)
            print(f"  pose: R saved  t={t}")

        # ── per-frame point cloud (local / camera frame) ──────────────────
        if local_pts_np is not None:
            pts_hw3 = local_pts_np[i]                       # (H, W, 3)
            xyz, rgb = points_and_colors_from_frame(
                pts_hw3, img_rgb, valid_mask=mask_valid
            )
            ply_path = os.path.join(per_frame_dir, f"{stem}.ply")
            save_ply(ply_path, xyz, rgb)

        # ── accumulate world-frame points for merged cloud ─────────────────
        if world_pts_np is not None:
            pts_hw3_w = world_pts_np[i]                     # (H, W, 3)
            xyz_w, rgb_w = points_and_colors_from_frame(
                pts_hw3_w, img_rgb, valid_mask=mask_valid
            )
            all_xyz.append(xyz_w)
            all_rgb.append(rgb_w)

    # ── merged point cloud ────────────────────────────────────────────────
    if all_xyz:
        merged_xyz = np.concatenate(all_xyz, axis=0)
        merged_rgb = np.concatenate(all_rgb, axis=0)
        save_ply(os.path.join(merged_dir, "merged.ply"), merged_xyz, merged_rgb)
        print(f"\n[pipeline] Merged cloud: {len(merged_xyz):,} points total.")

    # ── save all poses together ───────────────────────────────────────────
    if poses_np is not None:
        np.save(os.path.join(pose_dir, "all_poses.npy"), poses_np)
        np.save(
            os.path.join(pose_dir, "all_rotations.npy"),
            poses_np[:, :3, :3],
        )
        print(f"[pipeline] All poses ({S}) saved → {pose_dir}/all_poses.npy")

    print(f"\n✅ Done.  Results written to: {out_root}")


# =========================================================================
#  8.  Entry Point
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PanoVGGT Inference — depth, poses, point clouds"
    )
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
                   help="Use logarithmic scale for depth visualisation.")
    p.add_argument("--no_log_depth", dest="log_depth", action="store_false",
                   help="Disable logarithmic depth scale.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
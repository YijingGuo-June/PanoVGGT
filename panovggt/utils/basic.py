"""
panovggt.utils.basic
====================

General-purpose utilities for panoramic 3D reconstruction:

* Image loading with automatic 2:1 aspect-ratio + patch-aligned resizing
* Panoramic radial-depth visualisation
* Point-cloud / camera PLY export (Open3D)
* COLMAP-format camera export
* GLB scene construction (trimesh) for interactive viewers
"""

import os
import os.path as osp
import json
import math
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from PIL import Image

import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation


# =========================================================================
#  1.  Image Loading
# =========================================================================


def _panorama_target_size(
    orig_h: int,
    orig_w: int,
    patch_size: int = 14,
) -> Tuple[int, int]:
    """
    Return fixed panoramic size (518, 1036) for standard panorama resolution.
    
    This enforces a consistent 2:1 aspect ratio divisible by patch_size (14).
    - Height: 518 = 37 * 14
    - Width: 1036 = 74 * 14 = 2 * 518
    
    Args:
        orig_h: Original image height (unused, kept for API compatibility).
        orig_w: Original image width (unused, kept for API compatibility).
        patch_size: Spatial patch size (default 14). Must be 14 for this implementation.
    
    Returns:
        Tuple of (518, 1036) representing (height, width).
    """
    if patch_size != 14:
        raise ValueError(f"Only patch_size=14 is supported, got {patch_size}")
    
    return 518, 1036


def load_images_as_tensor(
    path: str = "data/truck",
    interval: int = 1,
    patch_size: int = 14,
    target_height: Optional[int] = None,
) -> torch.Tensor:
    """
    Load images from a directory (or ``.mp4`` video) and return a
    ``[N, 3, 518, 1036]`` float tensor in ``[0, 1]``.

    **Fixed Panoramic Resolution** — every frame is resized to exactly:
    - Height: **518 pixels** (37 patches × 14)
    - Width: **1036 pixels** (74 patches × 14)
    - Aspect ratio: **2:1** (width = 2 × height)

    Args:
        path: Directory of images **or** path to a ``.mp4`` video.
        interval: Keep every *interval*-th image / frame.
        patch_size: Must be 14 (other values will raise an error).
        target_height: Ignored (kept for backward compatibility).

    Returns:
        ``torch.Tensor`` of shape ``[N, 3, 518, 1036]``.
    """
    if patch_size != 14:
        raise ValueError(f"Only patch_size=14 is supported for 518×1036 resolution, got {patch_size}")
    
    sources: List[Image.Image] = []

    # ── 1. Collect PIL images ────────────────────────────────────────────
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        names = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
        for i in range(0, len(names), interval):
            try:
                sources.append(Image.open(osp.join(path, names[i])).convert("RGB"))
            except Exception as e:
                print(f"  skip {names[i]}: {e}")
    elif path.lower().endswith(".mp4"):
        print(f"Loading frames from video: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % interval == 0:
                sources.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            idx += 1
        cap.release()
    else:
        raise ValueError(f"Unsupported path (need directory or .mp4): {path}")

    if not sources:
        print("No images loaded.")
        return torch.empty(0)

    print(f"Loaded {len(sources)} image(s).")

    # ── 2. Fixed target size ─────────────────────────────────────────────
    TARGET_H, TARGET_W = 518, 1036
    print(f"  target size (H×W): {TARGET_H} × {TARGET_W}  "
          f"(patch_size={patch_size}, ratio={TARGET_W / TARGET_H:.2f})")

    # ── 3. Resize → tensor ───────────────────────────────────────────────
    to_tensor = transforms.ToTensor()
    tensors: List[torch.Tensor] = []

    for i, img in enumerate(sources):
        try:
            resized = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            tensors.append(to_tensor(resized))
            if i == 0 or i == len(sources) - 1:
                ow, oh = img.size
                print(f"  frame {i}: {ow}×{oh} → {TARGET_W}×{TARGET_H}")
        except Exception as e:
            print(f"  error on frame {i}: {e}")

    if not tensors:
        print("All frames failed to process.")
        return torch.empty(0)

    stacked = torch.stack(tensors, dim=0)
    print(f"  tensor shape: {stacked.shape}  (N, C, H, W)")
    return stacked


# def _panorama_target_size(
#     orig_h: int,
#     orig_w: int,
#     patch_size: int = 14,
# ) -> Tuple[int, int]:
#     """Return ``(H, W)`` that satisfies **2 : 1 aspect ratio** and is
#     divisible by *patch_size*.

#     The target height is chosen as the multiple of *patch_size* closest to the
#     original height.  Width is always ``2 * H``, which is automatically a
#     multiple of *patch_size* as well (since ``2 * k * patch_size`` is divisible
#     by *patch_size*).
#     """
#     target_h = max(round(orig_h / patch_size) * patch_size, patch_size)
#     target_w = 2 * target_h
#     return target_h, target_w


# def load_images_as_tensor(
#     path: str = "data/truck",
#     interval: int = 1,
#     patch_size: int = 14,
#     target_height: Optional[int] = None,
# ) -> torch.Tensor:
#     """Load images from a directory (or ``.mp4`` video) and return a
#     ``[N, 3, H, W]`` float tensor in ``[0, 1]``.

#     **Panoramic convention** — every frame is resized so that:

#     * the aspect ratio is exactly **2 : 1** (width = 2 × height), and
#     * both height and width are multiples of *patch_size* (default 14).

#     Args:
#         path: Directory of images **or** path to a ``.mp4`` video.
#         interval: Keep every *interval*-th image / frame.
#         patch_size: Spatial patch size of the vision backbone (default 14
#             for ViT).  Height and width will be rounded to the nearest
#             multiple.
#         target_height: If given, use this value (rounded to a multiple of
#             *patch_size*) instead of inferring from the first image.

#     Returns:
#         ``torch.Tensor`` of shape ``[N, 3, H, W]``.
#     """
#     sources: List[Image.Image] = []

#     # ── 1. collect PIL images ────────────────────────────────────────────
#     if osp.isdir(path):
#         print(f"Loading images from directory: {path}")
#         exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
#         names = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
#         for i in range(0, len(names), interval):
#             try:
#                 sources.append(Image.open(osp.join(path, names[i])).convert("RGB"))
#             except Exception as e:
#                 print(f"  skip {names[i]}: {e}")
#     elif path.lower().endswith(".mp4"):
#         print(f"Loading frames from video: {path}")
#         cap = cv2.VideoCapture(path)
#         if not cap.isOpened():
#             raise IOError(f"Cannot open video: {path}")
#         idx = 0
#         while True:
#             ok, frame = cap.read()
#             if not ok:
#                 break
#             if idx % interval == 0:
#                 sources.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#             idx += 1
#         cap.release()
#     else:
#         raise ValueError(f"Unsupported path (need directory or .mp4): {path}")

#     if not sources:
#         print("No images loaded.")
#         return torch.empty(0)

#     print(f"Loaded {len(sources)} image(s).")

#     # ── 2. compute target size ───────────────────────────────────────────
#     if target_height is not None:
#         th = max(round(target_height / patch_size) * patch_size, patch_size)
#         tw = 2 * th
#     else:
#         first_w, first_h = sources[0].size          # PIL: (W, H)
#         th, tw = _panorama_target_size(first_h, first_w, patch_size)

#     print(f"  target size (H×W): {th} × {tw}  "
#           f"(patch_size={patch_size}, ratio={tw / th:.2f})")

#     # ── 3. resize → tensor ───────────────────────────────────────────────
#     to_tensor = transforms.ToTensor()
#     tensors: List[torch.Tensor] = []

#     for i, img in enumerate(sources):
#         try:
#             resized = img.resize((tw, th), Image.Resampling.LANCZOS)
#             tensors.append(to_tensor(resized))
#             if i == 0 or i == len(sources) - 1:
#                 ow, oh = img.size
#                 print(f"  frame {i}: {ow}×{oh} → {tw}×{th}")
#         except Exception as e:
#             print(f"  error on frame {i}: {e}")

#     if not tensors:
#         print("All frames failed to process.")
#         return torch.empty(0)

#     stacked = torch.stack(tensors, dim=0)
#     print(f"  tensor shape: {stacked.shape}  (N, C, H, W)")
#     return stacked


# =========================================================================
#  2.  Panoramic Depth Visualisation
# =========================================================================

def visualize_panorama_depth(
    depth: np.ndarray,
    colormap: str = "turbo",
    invalid_mask: Optional[np.ndarray] = None,
    use_log_scale: bool = True,
    depth_min_percentile: float = 2.0,
    depth_max_percentile: float = 98.0,
) -> np.ndarray:
    """Colour-map radial depth for panoramic images.

    Args:
        depth: ``(N, H, W)`` or ``(H, W)`` depth array.
        colormap: Matplotlib colourmap name.
        invalid_mask: Optional boolean mask marking pixels to ignore.
        use_log_scale: Apply log scaling before colour mapping.
        depth_min_percentile / depth_max_percentile: Clipping percentiles.

    Returns:
        ``uint8`` RGB array matching the leading shape of *depth*.
    """
    single = depth.ndim == 2
    if single:
        depth = depth[None]

    cmap = plt.get_cmap(colormap)
    frames = []

    for i in range(depth.shape[0]):
        d = depth[i].copy()
        valid = np.isfinite(d) & (d > 0) & (d < 1e6)
        if invalid_mask is not None:
            m = invalid_mask[i] if invalid_mask.ndim == 3 else invalid_mask
            valid &= ~m

        if valid.any():
            lo = np.percentile(d[valid], depth_min_percentile)
            hi = np.percentile(d[valid], depth_max_percentile)
            dc = np.clip(d, lo, hi)
            if use_log_scale and lo > 0:
                llo, lhi = np.log(lo + 1e-6), np.log(hi + 1e-6)
                norm = (np.log(dc + 1e-6) - llo) / max(lhi - llo, 1e-6)
            else:
                norm = (dc - lo) / max(hi - lo, 1e-6)
            norm = np.clip(norm, 0, 1)
        else:
            norm = np.zeros_like(d)

        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        rgb[~valid] = 0
        frames.append(rgb)

    result = np.stack(frames)
    return result[0] if single else result


def save_panorama_depth_visualizations(
    depths: np.ndarray,
    save_dir: str,
    prefix: str = "depth",
    colormap: str = "turbo",
    use_log_scale: bool = True,
) -> List[str]:
    """Save per-frame depth colour-maps as PNG files."""
    os.makedirs(save_dir, exist_ok=True)
    coloured = visualize_panorama_depth(depths, colormap=colormap, use_log_scale=use_log_scale)
    paths = []
    for i in range(len(coloured)):
        p = os.path.join(save_dir, f"{prefix}_{i:04d}.png")
        cv2.imwrite(p, cv2.cvtColor(coloured[i], cv2.COLOR_RGB2BGR))
        paths.append(p)
    return paths


def create_panorama_depth_comparison(
    images: np.ndarray,
    depths: np.ndarray,
    colormap: str = "turbo",
    use_log_scale: bool = True,
) -> List[np.ndarray]:
    """Stack each RGB frame above its depth colour-map (vertical concat)."""
    coloured = visualize_panorama_depth(depths, colormap=colormap, use_log_scale=use_log_scale)
    imgs = images.copy()
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).astype(np.uint8) if imgs.max() <= 1.0 else imgs.astype(np.uint8)
    out = []
    for i in range(len(imgs)):
        h, w = imgs[i].shape[:2]
        dc = cv2.resize(coloured[i], (w, h)) if coloured[i].shape[:2] != (h, w) else coloured[i]
        out.append(np.concatenate([imgs[i], dc], axis=0))
    return out


def create_panorama_depth_grid(
    images: np.ndarray,
    depths: np.ndarray,
    colormap: str = "turbo",
    use_log_scale: bool = True,
    max_frames: int = 8,
) -> np.ndarray:
    """Create a compact ``[RGB | Depth]`` grid, one row per frame."""
    n = min(len(images), max_frames)
    coloured = visualize_panorama_depth(depths[:n], colormap=colormap, use_log_scale=use_log_scale)
    imgs = images[:n].copy()
    if imgs.dtype != np.uint8:
        imgs = (imgs * 255).astype(np.uint8) if imgs.max() <= 1.0 else imgs.astype(np.uint8)
    sf = 0.5 if imgs.shape[2] > 1000 else 1.0
    rows = []
    for i in range(n):
        h, w = imgs[i].shape[:2]
        nh, nw = int(h * sf), int(w * sf)
        rows.append(np.concatenate([cv2.resize(imgs[i], (nw, nh)),
                                    cv2.resize(coloured[i], (nw, nh))], axis=1))
    return np.concatenate(rows, axis=0)


# =========================================================================
#  3.  Point-Cloud & Camera PLY Export  (Open3D)
# =========================================================================

def _camera_frustum_mesh(
    pose_c2w: np.ndarray,
    scale: float = 0.1,
    aspect: float = 1.0,
    color: Tuple[float, ...] = (1.0, 0.0, 0.0),
) -> o3d.geometry.TriangleMesh:
    """Tiny triangulated camera-frustum mesh in world coordinates."""
    h, w, d = scale, scale * aspect, scale * 1.5
    verts = np.array([[0, 0, 0],
                      [-w / 2, -h / 2, d], [w / 2, -h / 2, d],
                      [w / 2, h / 2, d], [-w / 2, h / 2, d]])
    verts = (pose_c2w[:3, :3] @ verts.T).T + pose_c2w[:3, 3]
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4],
                      [0, 4, 1], [1, 2, 3], [1, 3, 4]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(faces))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


_VIS_SCRIPT = r'''#!/usr/bin/env python3
"""Open3D viewer for exported point cloud + cameras.

Usage
-----
    python visualize.py                       # interactive
    python visualize.py --screenshot fig.png  # headless render
"""
import argparse, json, os
import numpy as np
import open3d as o3d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cameras", action="store_true")
    ap.add_argument("--screenshot", type=str, default=None)
    ap.add_argument("--point-size", type=float, default=1.0)
    ap.add_argument("--bg", type=float, nargs=3, default=[1, 1, 1])
    args = ap.parse_args()

    d = os.path.dirname(os.path.abspath(__file__))
    geoms = [o3d.io.read_point_cloud(os.path.join(d, "pointcloud.ply"))]

    if not args.no_cameras:
        fp = os.path.join(d, "camera_frustums.ply")
        if os.path.exists(fp):
            geoms.append(o3d.io.read_triangle_mesh(fp))

    info_p = os.path.join(d, "camera_info.json")
    if os.path.exists(info_p):
        with open(info_p) as f:
            ci = json.load(f)
        pts = np.array([c["center"] for c in ci["cameras"]])
        if len(pts) > 1:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines  = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(pts)-1)])
            ls.colors = o3d.utility.Vector3dVector([[.3,.3,.3]]*(len(pts)-1))
            geoms.append(ls)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.array(args.bg)
    opt.point_size = args.point_size

    if args.screenshot:
        vis.poll_events(); vis.update_renderer()
        vis.capture_screen_image(args.screenshot)
        print(f"Saved {args.screenshot}")
        vis.destroy_window()
    else:
        vis.run(); vis.destroy_window()

if __name__ == "__main__":
    main()
'''


def save_pointcloud_and_cameras(
    predictions: dict,
    save_dir: str,
    camera_scale: float = 0.1,
    include_colors: bool = True,
    save_separate: bool = True,
    save_combined: bool = True,
) -> Dict[str, str]:
    """Export point cloud + camera frustums as PLY files.

    Expected keys in *predictions*: ``points``, ``images``, ``camera_poses``.

    Returns:
        Dict mapping descriptive names to file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    saved: Dict[str, str] = {}

    pts = predictions["points"].reshape(-1, 3)
    images = predictions["images"]
    poses = predictions["camera_poses"]

    # colours
    if images.ndim == 4 and images.shape[-1] == 3:
        colours = images.reshape(-1, 3).copy()
    else:
        colours = np.full((len(pts), 3), 0.5)
    if colours.max() > 1.0:
        colours = colours / 255.0

    # keep only finite points
    mask = np.isfinite(pts).all(axis=1)
    pts, colours = pts[mask], colours[mask]

    # scene scale → camera frustum sizing
    if len(pts) > 0:
        lo, hi = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
        scene_scale = np.linalg.norm(hi - lo)
    else:
        scene_scale = 1.0
    cam_s = camera_scale * scene_scale

    # point cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if include_colors:
        pcd.colors = o3d.utility.Vector3dVector(colours)

    # camera frustums
    cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
    n_cam = len(poses)
    centres, cam_colours, frustums = [], [], []
    for i in range(n_cam):
        t = i / max(n_cam - 1, 1)
        c = (np.array(cmap(t)[:3]) * 0.4 + 0.6).tolist()       # pastel
        centres.append(poses[i][:3, 3])
        cam_colours.append(c)
        frustums.append(_camera_frustum_mesh(poses[i], scale=cam_s * 0.5, color=c))

    centres = np.asarray(centres)
    cam_colours = np.asarray(cam_colours)

    # ── write files ──────────────────────────────────────────────────────
    if save_separate:
        p = os.path.join(save_dir, "pointcloud.ply")
        o3d.io.write_point_cloud(p, pcd);  saved["pointcloud"] = p

        cp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centres))
        cp.colors = o3d.utility.Vector3dVector(cam_colours)
        p = os.path.join(save_dir, "camera_centers.ply")
        o3d.io.write_point_cloud(p, cp);  saved["camera_centers"] = p

        if frustums:
            merged = frustums[0]
            for f in frustums[1:]:
                merged += f
            p = os.path.join(save_dir, "camera_frustums.ply")
            o3d.io.write_triangle_mesh(p, merged);  saved["camera_frustums"] = p

    if save_combined:
        comb = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(np.vstack([pts, centres])))
        comb.colors = o3d.utility.Vector3dVector(np.vstack([colours, cam_colours]))
        p = os.path.join(save_dir, "pointcloud_with_cameras.ply")
        o3d.io.write_point_cloud(p, comb);  saved["combined"] = p

    # camera info JSON
    info: dict = {"num_cameras": n_cam, "scene_scale": float(scene_scale),
                  "camera_scale": float(cam_s), "cameras": []}
    for i in range(n_cam):
        info["cameras"].append({"id": i,
                                "pose_c2w": poses[i].tolist(),
                                "center": centres[i].tolist(),
                                "color": cam_colours[i].tolist()})
    p = os.path.join(save_dir, "camera_info.json")
    with open(p, "w") as fh:
        json.dump(info, fh, indent=2)
    saved["camera_info"] = p

    # standalone vis script
    p = os.path.join(save_dir, "visualize.py")
    with open(p, "w") as fh:
        fh.write(_VIS_SCRIPT)
    saved["vis_script"] = p

    return saved


def save_cameras_as_colmap(
    camera_poses: np.ndarray,
    save_dir: str,
    image_names: Optional[List[str]] = None,
) -> str:
    """Write camera poses in minimal COLMAP text format."""
    os.makedirs(save_dir, exist_ok=True)
    n = len(camera_poses)
    if image_names is None:
        image_names = [f"image_{i:04d}.jpg" for i in range(n)]

    with open(os.path.join(save_dir, "images.txt"), "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(n):
            w2c = np.linalg.inv(camera_poses[i])
            q = Rotation.from_matrix(w2c[:3, :3]).as_quat()       # (x,y,z,w)
            t = w2c[:3, 3]
            f.write(f"{i+1} {q[3]} {q[0]} {q[1]} {q[2]} "
                    f"{t[0]} {t[1]} {t[2]} 1 {image_names[i]}\n\n")

    with open(os.path.join(save_dir, "cameras.txt"), "w") as f:
        f.write("1 PINHOLE 1024 512 512 512 512 256\n")

    open(os.path.join(save_dir, "points3D.txt"), "w").close()
    return save_dir


# =========================================================================
#  4.  GLB Scene Builder  (for Gradio Model3D viewer)
# =========================================================================

def _opengl_flip() -> np.ndarray:
    m = np.eye(4);  m[1, 1] = m[2, 2] = -1;  return m


def _xform_pts(T: np.ndarray, pts: np.ndarray, dim: int = None) -> np.ndarray:
    dim = dim or pts.shape[-1]
    Tt = T.swapaxes(-1, -2)
    return (pts @ Tt[..., :-1, :] + Tt[..., -1:, :])[..., :dim]


def _cone_faces(cone: trimesh.Trimesh) -> np.ndarray:
    nv = len(cone.vertices)
    faces = []
    for f in cone.faces:
        if 0 in f:
            continue
        v1, v2, v3 = f
        for off in (nv, 2 * nv):
            faces += [(v1, v2, v2 + off),
                      (v1, v1 + off, v3),
                      (v3 + off, v2, v3)]
    faces += [(c, b, a) for a, b, c in faces]          # back-faces
    return np.array(faces)


def _add_glb_camera(scene: trimesh.Scene, c2w: np.ndarray,
                    color: tuple, scale: float):
    w, h = scale * 0.05, scale * 0.1
    r45 = np.eye(4)
    r45[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    r45[2, 3] = -h
    xf = c2w @ _opengl_flip() @ r45
    cone = trimesh.creation.cone(w, h, sections=4)
    r2 = np.eye(4)
    r2[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()
    verts = np.concatenate([cone.vertices, 0.95 * cone.vertices,
                            _xform_pts(r2, cone.vertices)])
    verts = _xform_pts(xf, verts)
    mesh = trimesh.Trimesh(vertices=verts, faces=_cone_faces(cone))
    mesh.visual.face_colors[:, :3] = color
    scene.add_geometry(mesh)


def predictions_to_glb(
    predictions: dict,
    filter_by_frames: str = "All",
    show_cam: bool = True,
) -> trimesh.Scene:
    """Build a ``trimesh.Scene`` from model predictions (exportable as GLB).

    Expected keys: ``points``, ``images``, ``camera_poses``.
    """
    sel = None
    if filter_by_frames not in ("all", "All"):
        try:
            sel = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pts = predictions["points"]
    images = predictions["images"]
    poses = predictions["camera_poses"]

    if sel is not None:
        pts, images, poses = pts[sel:sel+1], images[sel:sel+1], poses[sel:sel+1]

    verts = pts.reshape(-1, 3)
    rgb = (images.reshape(-1, 3) * 255).astype(np.uint8)
    valid = np.isfinite(verts).all(axis=1)
    verts, rgb = verts[valid], rgb[valid]

    if verts.size == 0:
        verts = np.array([[1.0, 0.0, 0.0]])
        rgb = np.array([[255, 255, 255]], dtype=np.uint8)

    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=verts, colors=rgb))

    if show_cam:
        cmap = matplotlib.colormaps.get_cmap("gist_rainbow")
        for i, pose in enumerate(poses):
            rgba = cmap(i / max(len(poses), 1))
            _add_glb_camera(scene, pose,
                            tuple(int(255 * x) for x in rgba[:3]), 1.0)

    rot = np.eye(4)
    rot[:3, :3] = (Rotation.from_euler("y", 100, degrees=True).as_matrix()
                   @ Rotation.from_euler("x", 155, degrees=True).as_matrix())
    scene.apply_transform(rot)
    return scene
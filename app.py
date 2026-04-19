#!/usr/bin/env python3
"""
PanoVGGT — Panoramic Visual Geometry Grounded Transformer
Gradio Interactive Demo
"""

import os

# ── Strip proxy settings ──────────────────────────────────────────────────
for _var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
             "ALL_PROXY", "all_proxy"):
    os.environ.pop(_var, None)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gc
import cv2
import glob
import time
import shutil
import argparse
import numpy as np
from datetime import datetime
from typing import List, Tuple

import torch
from omegaconf import OmegaConf
import gradio as gr

from panovggt.models.panovggt_model import PanoVGGTModel
from panovggt.utils.basic import (
    load_images_as_tensor,
    save_panorama_depth_visualizations,
    create_panorama_depth_comparison,
    create_panorama_depth_grid,
    save_pointcloud_and_cameras,
    save_cameras_as_colmap,
    predictions_to_glb,
)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_EXAMPLE_DIR = "examples"


# =========================================================================
#  Helper: locate images for a scene (supports both layouts)
# =========================================================================

def _scene_image_dir(scene_name: str) -> str:
    """
    Return the directory that actually contains the images for a scene.

    Supports two layouts automatically:
      Layout A — examples/<scene>/images/*.jpg   (preferred)
      Layout B — examples/<scene>/*.jpg
    """
    base = os.path.join(_EXAMPLE_DIR, scene_name)
    sub  = os.path.join(base, "images")
    if os.path.isdir(sub) and any(
        os.path.splitext(f)[1].lower() in _IMG_EXTS for f in os.listdir(sub)
    ):
        return sub      # Layout A
    return base         # Layout B (images directly in scene folder)


def _collect_example_images(scene_name: str) -> List[str]:
    """Return sorted absolute image paths for a given example scene."""
    img_dir = _scene_image_dir(scene_name)
    paths = sorted(
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    )
    if not paths:
        raise ValueError(f"No images found in: {img_dir}")
    return paths


def _discover_scenes() -> List[str]:
    """
    Scan examples/ and return scene names that contain at least one image
    (either directly or inside an images/ sub-directory).
    """
    scenes: List[str] = []
    if not os.path.isdir(_EXAMPLE_DIR):
        return scenes
    for name in sorted(os.listdir(_EXAMPLE_DIR)):
        candidate = os.path.join(_EXAMPLE_DIR, name)
        if not os.path.isdir(candidate):
            continue
        try:
            imgs = _collect_example_images(name)
            if imgs:
                scenes.append(name)
        except ValueError:
            pass
    return scenes


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
#  2.  Inference
# =========================================================================

def run_model(
    target_dir: str,
    model: PanoVGGTModel,
    depth_colormap: str = "turbo",
    use_log_scale: bool = True,
    save_ply: bool = True,
    camera_scale: float = 0.05,
) -> dict:
    device = next(model.parameters()).device
    imgs = load_images_as_tensor(
        os.path.join(target_dir, "images"), interval=1
    ).to(device)
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if len(image_names) == 0:
        raise ValueError("No images found — check your upload.")

    print(f"[PanoVGGT] Running inference on {len(image_names)} panoramic frames …")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        preds = model(imgs[None])

    preds["images"] = imgs[None].permute(0, 1, 3, 4, 2)
    if "world_points" in preds:
        preds["points"] = preds["world_points"]

    radial = None
    if "local_points" in preds and preds["local_points"] is not None:
        lp = preds["local_points"]
        radial = torch.norm(lp, dim=-1)
        preds["z_depth"] = lp[..., 2].clone()
    elif "depth" in preds and preds["depth"] is not None:
        radial = preds["depth"].clone()
    elif "points" in preds and "camera_poses" in preds:
        cp = preds["camera_poses"][..., :3, 3][:, :, None, None, :]
        radial = torch.norm(preds["points"] - cp, dim=-1)
    if radial is not None:
        preds["radial_depth"] = radial
        preds["depth"] = radial

    out: dict = {}
    for k, v in preds.items():
        if isinstance(v, torch.Tensor):
            out[k] = (v.float() if v.dtype == torch.bfloat16 else v).cpu().numpy().squeeze(0)
        else:
            out[k] = v

    if "depth" in out and out["depth"] is not None:
        d = out["depth"]
        out["depth_viz_paths"] = save_panorama_depth_visualizations(
            d, os.path.join(target_dir, "depth_visualization"),
            colormap=depth_colormap, use_log_scale=use_log_scale)

        if "images" in out and out["images"].ndim == 4 and out["images"].shape[-1] == 3:
            cdir = os.path.join(target_dir, "depth_comparison")
            os.makedirs(cdir, exist_ok=True)
            comps = create_panorama_depth_comparison(
                out["images"], d, colormap=depth_colormap, use_log_scale=use_log_scale)
            cp = []
            for i, c in enumerate(comps):
                p = os.path.join(cdir, f"comparison_{i:04d}.png")
                cv2.imwrite(p, cv2.cvtColor(c, cv2.COLOR_RGB2BGR))
                cp.append(p)
            out["comparison_paths"] = cp

            grid = create_panorama_depth_grid(
                out["images"], d, colormap=depth_colormap, use_log_scale=use_log_scale)
            cv2.imwrite(os.path.join(target_dir, "depth_comparison_grid.png"),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    if save_ply and "points" in out and "camera_poses" in out:
        out["ply_files"] = save_pointcloud_and_cameras(
            out, os.path.join(target_dir, "ply_output"),
            camera_scale=camera_scale)
        save_cameras_as_colmap(
            out["camera_poses"],
            os.path.join(target_dir, "colmap_format"),
            [os.path.basename(p) for p in image_names])

    out.pop("local_points", None)
    torch.cuda.empty_cache()
    return out


# =========================================================================
#  3.  Gradio Callbacks
# =========================================================================

def handle_uploads(input_video, input_images, interval: int = -1) -> Tuple[str, List[str]]:
    gc.collect(); torch.cuda.empty_cache()
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tdir = f"panovggt_session_{ts}"
    idir = os.path.join(tdir, "images")
    os.makedirs(idir, exist_ok=True)
    paths: List[str] = []

    if input_images is not None:
        sel = input_images[::interval] if interval and interval > 0 else input_images
        for fd in sel:
            fp  = fd["name"] if isinstance(fd, dict) and "name" in fd else fd
            dst = os.path.join(idir, os.path.basename(fp))
            shutil.copy(fp, dst)
            paths.append(dst)

    if input_video is not None:
        vp  = (input_video["name"]
               if isinstance(input_video, dict) and "name" in input_video
               else input_video)
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = interval if (interval and interval > 0) else max(int(fps), 1)
        cnt = idx = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            cnt += 1
            if cnt % step == 0:
                p = os.path.join(idir, f"{idx:06d}.png")
                cv2.imwrite(p, frame)
                paths.append(p); idx += 1
        cap.release()

    return tdir, sorted(paths)


def update_gallery_on_upload(input_video, input_images, interval=-1):
    if not input_video and not input_images:
        return None, None, None, "Please upload a video or image set."
    tdir, paths = handle_uploads(input_video, input_images, interval=interval)
    msg = (f"✅ Upload complete — **{len(paths)}** frame(s) extracted.\n\n"
           f"Press **Run Reconstruction** to start PanoVGGT inference.")
    return None, tdir, paths, msg


def gradio_reconstruct(
    target_dir, frame_filter="All", show_cam=True,
    depth_colormap="turbo", use_log_scale=True,
    save_ply=True, camera_scale=0.05,
):
    if not target_dir or not os.path.isdir(target_dir):
        return (None,
                "⚠️ No valid session directory found. Please upload media first.",
                None, None, None, None)

    gc.collect(); torch.cuda.empty_cache()
    t0 = time.time()

    idir  = os.path.join(target_dir, "images")
    files = sorted(os.listdir(idir)) if os.path.isdir(idir) else []
    choices = ["All"] + [f"{i}: {f}" for i, f in enumerate(files)]

    with torch.no_grad():
        preds = run_model(
            target_dir, model,
            depth_colormap=depth_colormap,
            use_log_scale=use_log_scale,
            save_ply=save_ply,
            camera_scale=camera_scale,
        )

    saveable = {k: v for k, v in preds.items()
                if k not in ("depth_viz_paths", "comparison_paths", "ply_files", "colmap_dir")}
    np.savez(os.path.join(target_dir, "predictions.npz"), **saveable)

    frame_filter = frame_filter or "All"
    glb = os.path.join(target_dir,
                       f"panovggt_{frame_filter.replace(' ', '_')}_cam{show_cam}.glb")
    predictions_to_glb(preds, frame_filter, show_cam).export(file_obj=glb)

    depth_paths = preds.get("depth_viz_paths", [])
    grid_path   = os.path.join(target_dir, "depth_comparison_grid.png")
    ply_path    = (preds.get("ply_files") or {}).get("pointcloud")

    elapsed = time.time() - t0
    msg = (f"✅ **Reconstruction complete!**\n\n"
           f"- Frames processed : **{len(files)}**\n"
           f"- Total time       : **{elapsed:.1f} s**\n"
           f"- Session directory: `{target_dir}`")

    del preds; gc.collect(); torch.cuda.empty_cache()
    return (glb, msg,
            gr.Dropdown(choices=choices, value=frame_filter, interactive=True),
            depth_paths,
            grid_path if os.path.exists(grid_path) else None,
            ply_path)


def update_visualization(target_dir, frame_filter, show_cam, is_example):
    if is_example == "True" or not target_dir or not os.path.isdir(target_dir):
        return None, "Please run **Run Reconstruction** first."
    npz = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(npz):
        return None, "Please run **Run Reconstruction** first."
    loaded = np.load(npz)
    preds  = {k: np.array(loaded[k]) for k in loaded.files}
    glb = os.path.join(target_dir,
                       f"panovggt_{frame_filter.replace(' ', '_')}_cam{show_cam}.glb")
    if not os.path.exists(glb):
        predictions_to_glb(preds, frame_filter, show_cam).export(file_obj=glb)
    return glb, "🔄 Visualisation updated."


def example_pipeline(
    scene_name: str,
    show_cam: bool,
    depth_colormap: str,
    use_log_scale: bool,
    save_ply: bool,
    camera_scale: float,
):
    """
    End-to-end pipeline for built-in example scenes.
    Copies images from examples/<scene_name>/ (or examples/<scene_name>/images/)
    into a fresh session directory, then runs reconstruction.
    """
    print(f"[PanoVGGT] Loading example scene: '{scene_name}'")
    src_paths = _collect_example_images(scene_name)

    # ── fresh session ─────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tdir = f"panovggt_session_{ts}"
    idir = os.path.join(tdir, "images")
    os.makedirs(idir, exist_ok=True)

    dst_paths: List[str] = []
    for src in src_paths:
        dst = os.path.join(idir, os.path.basename(src))
        shutil.copy(src, dst)
        dst_paths.append(dst)

    print(f"[PanoVGGT] Example '{scene_name}' — {len(dst_paths)} images copied.")

    glb, msg, dd, dv, dg, pp = gradio_reconstruct(
        tdir, "All", show_cam,
        depth_colormap, use_log_scale, save_ply, camera_scale)

    return glb, msg, tdir, dd, dst_paths, dv, dg, pp


# =========================================================================
#  4.  CSS & HTML
# =========================================================================

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;700&display=swap');

@keyframes bg-shift {
  0%,100% { background-position: 0% 50%; }
  50%      { background-position: 100% 50%; }
}
@keyframes title-glow {
  0%,100% { text-shadow: 0 0 14px #34d399, 0 0 28px #059669; }
  50%      { text-shadow: 0 0  6px #34d399, 0 0 12px #059669; }
}
@keyframes card-pulse {
  0%,100% { box-shadow: 0 4px 24px rgba(52,211,153,.12); }
  50%      { box-shadow: 0 4px 36px rgba(52,211,153,.28); }
}

.gradio-container {
  font-family: 'Space Grotesk', sans-serif !important;
  background: linear-gradient(-45deg,#020c07,#071a0f,#051510,#0a1f15);
  background-size: 400% 400%;
  animation: bg-shift 28s ease infinite;
  color: #a7f3d0 !important;
  min-height: 100vh;
}
.gradio-container,.gr-label label,.gr-input,
input,textarea,.gr-check-radio label,
.gr-block label,.svelte-1ed2p3z { color:#d1fae5 !important; }
.gr-markdown p,.gr-markdown li,.gr-markdown h3,
.gr-markdown h4 { color:#a7f3d0 !important; }
.gr-markdown strong { color:#6ee7b7 !important; }
.gr-markdown code {
  background:rgba(6,78,59,0.45) !important; color:#6ee7b7 !important;
  border-radius:4px; padding:1px 5px;
}
.gr-markdown pre {
  background:rgba(2,20,10,0.70) !important;
  border:1px solid rgba(52,211,153,0.25) !important;
  border-radius:10px; padding:14px;
}
thead th { color:#fff !important; background:#064e3b !important; }
tbody td  { color:#d1fae5 !important; }

.gr-block.gr-group {
  background:rgba(2,18,10,0.65) !important;
  backdrop-filter:blur(14px);
  border:1px solid rgba(52,211,153,0.22) !important;
  border-radius:18px !important;
  animation:card-pulse 7s ease-in-out infinite;
  padding:20px !important;
  transition:border-color 0.3s,box-shadow 0.3s;
}
.gr-block.gr-group:hover {
  border-color:rgba(52,211,153,0.50) !important;
  box-shadow:0 0 36px rgba(52,211,153,0.22) !important;
}

.gr-button {
  background:linear-gradient(135deg,#065f46,#059669,#10b981) !important;
  background-size:200% auto !important; color:#ecfdf5 !important;
  font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important;
  font-size:0.93rem !important; letter-spacing:0.07em !important;
  text-transform:uppercase !important; border:none !important;
  border-radius:12px !important;
  box-shadow:0 4px 18px rgba(5,150,105,0.45) !important;
  transition:all 0.3s ease !important;
}
.gr-button:hover {
  background-position:right center !important;
  box-shadow:0 6px 28px rgba(16,185,129,0.60) !important;
  transform:translateY(-2px) scale(1.02) !important;
}

.status-box {
  background:rgba(6,78,59,0.18) !important;
  border:1px solid rgba(52,211,153,0.30) !important;
  border-radius:12px !important; padding:14px 18px !important;
  font-family:'JetBrains Mono',monospace !important;
  font-size:0.87rem !important; color:#6ee7b7 !important; min-height:54px;
}

.section-title {
  font-family:'Space Grotesk',sans-serif; font-size:0.80rem; font-weight:700;
  color:#34d399 !important; letter-spacing:0.12em; text-transform:uppercase;
  border-bottom:1px solid rgba(52,211,153,0.25); padding-bottom:7px; margin:0 0 14px;
}

select,.gr-dropdown select {
  background:rgba(6,78,59,0.45) !important; color:#d1fae5 !important;
  border:1px solid rgba(52,211,153,0.30) !important; border-radius:8px !important;
}
input[type=range]    { accent-color:#10b981; }
input[type=checkbox] { accent-color:#10b981; }
input[type=number]   {
  background:rgba(6,78,59,0.35) !important; color:#d1fae5 !important;
  border:1px solid rgba(52,211,153,0.25) !important; border-radius:8px !important;
}
.gr-gallery  { border:1px solid rgba(52,211,153,0.18) !important; border-radius:12px !important; background:rgba(2,18,10,0.40) !important; }
.gr-model3d  { border:1px solid rgba(52,211,153,0.28) !important; border-radius:16px !important; background:#020c07 !important; }
.gr-accordion{ background:rgba(6,78,59,0.22) !important; border:1px solid rgba(52,211,153,0.18) !important; border-radius:10px !important; }

/* ── Example scene cover cards ── */
.scene-grid {
  display:flex; flex-wrap:wrap; gap:12px;
  justify-content:flex-start; margin-bottom:16px;
}
.scene-card {
  flex:0 0 150px; border-radius:12px; overflow:hidden;
  border:1px solid rgba(52,211,153,0.25);
  background:rgba(2,18,10,0.55);
  transition:transform 0.22s,box-shadow 0.22s,border-color 0.22s;
}
.scene-card:hover {
  transform:translateY(-4px) scale(1.03);
  box-shadow:0 6px 24px rgba(52,211,153,0.30);
  border-color:rgba(52,211,153,0.55);
}
.scene-card img { width:100%; height:95px; object-fit:cover; display:block; }
.scene-card-label {
  padding:5px 8px;
  font-family:'Space Grotesk',sans-serif; font-size:0.76rem; font-weight:700;
  color:#6ee7b7 !important; letter-spacing:0.04em; text-align:center;
  background:rgba(6,78,59,0.50);
}
"""

_HEADER_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700;900&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
  .pvgt-header{text-align:center;padding:36px 24px 16px;}
  .pvgt-badge-row{display:flex;justify-content:center;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:22px;}
  .pvgt-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 16px;border-radius:999px;font-family:'Space Grotesk',sans-serif;font-size:0.76rem;font-weight:700;letter-spacing:0.06em;text-decoration:none!important;transition:transform 0.2s,box-shadow 0.2s;}
  .pvgt-badge:hover{transform:translateY(-2px);box-shadow:0 4px 16px rgba(52,211,153,0.40);}
  .pvgt-badge.github  {background:rgba(6,78,59,0.65); border:1px solid rgba(52,211,153,0.50);color:#6ee7b7!important;}
  .pvgt-badge.paper   {background:rgba(5,150,105,0.20);border:1px solid rgba(16,185,129,0.45);color:#a7f3d0!important;}
  .pvgt-badge.weights {background:rgba(4,120,87,0.28); border:1px solid rgba(52,211,153,0.38);color:#d1fae5!important;}
  .pvgt-badge.dataset {background:rgba(2,100,70,0.30); border:1px solid rgba(52,211,153,0.42);color:#a7f3d0!important;}
  .pvgt-title{font-family:'Space Grotesk',sans-serif;font-size:clamp(2rem,5vw,3.4rem);font-weight:900;letter-spacing:-0.02em;line-height:1.1;margin:0 0 6px;background:linear-gradient(120deg,#34d399 0%,#10b981 35%,#6ee7b7 65%,#059669 100%);-webkit-background-clip:text;background-clip:text;color:transparent!important;animation:title-glow 4.5s ease-in-out infinite;}
  .pvgt-fullname{font-family:'Space Grotesk',sans-serif;font-size:clamp(0.78rem,1.6vw,0.98rem);font-weight:500;color:#6ee7b7!important;letter-spacing:0.06em;margin:0 0 22px;}
  .pvgt-pills{display:flex;flex-wrap:wrap;gap:9px;justify-content:center;margin-bottom:22px;}
  .pvgt-pill{display:inline-flex;align-items:center;gap:5px;padding:5px 15px;border-radius:999px;background:rgba(4,120,87,0.28);border:1px solid rgba(52,211,153,0.28);font-family:'Space Grotesk',sans-serif;font-size:0.78rem;font-weight:600;color:#6ee7b7!important;letter-spacing:0.03em;}
  .pvgt-desc{max-width:800px;margin:0 auto 26px;background:rgba(6,78,59,0.16);border:1px solid rgba(52,211,153,0.20);border-radius:16px;padding:20px 28px;text-align:left;}
  .pvgt-desc p{font-family:'Space Grotesk',sans-serif;font-size:0.95rem;color:#a7f3d0!important;line-height:1.75;margin:0 0 10px;}
  .pvgt-desc p:last-child{margin-bottom:0;} .pvgt-desc strong{color:#34d399!important;}
  .pvgt-steps{display:flex;gap:14px;flex-wrap:wrap;justify-content:center;max-width:880px;margin:0 auto 10px;}
  .pvgt-step{flex:1 1 175px;background:rgba(6,78,59,0.18);border:1px solid rgba(52,211,153,0.20);border-radius:14px;padding:18px 16px;text-align:center;transition:border-color 0.25s,box-shadow 0.25s;}
  .pvgt-step:hover{border-color:rgba(52,211,153,0.45);box-shadow:0 0 20px rgba(52,211,153,0.18);}
  .pvgt-step-num{width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#065f46,#10b981);color:#ecfdf5!important;font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1rem;display:flex;align-items:center;justify-content:center;margin:0 auto 10px;box-shadow:0 0 14px rgba(16,185,129,0.55);}
  .pvgt-step-title{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:0.88rem;color:#34d399!important;margin-bottom:5px;letter-spacing:0.04em;}
  .pvgt-step-desc{font-family:'Space Grotesk',sans-serif;font-size:0.80rem;color:#a7f3d0!important;line-height:1.5;}
  .pvgt-divider{border:none;border-top:1px solid rgba(52,211,153,0.16);margin:28px auto 0;max-width:880px;}
</style>
<div class="pvgt-header">
  <div class="pvgt-badge-row">
    <a class="pvgt-badge github"
       href="https://github.com/YijingGuo-June/PanoVGGT"
       target="_blank">🐙 GitHub</a>
    <a class="pvgt-badge paper"
       href="https://arxiv.org/pdf/2603.17571"
       target="_blank">📄 arXiv Paper</a>
    <a class="pvgt-badge dataset"
       href="https://huggingface.co/datasets/YijingGuo/PanoCity"
       target="_blank">🤗 PanoCity Dataset</a>
  </div>
  <h1 class="pvgt-title">PanoVGGT</h1>
  <p class="pvgt-fullname">Panoramic Visual Geometry Grounded Transformer</p>
  <div class="pvgt-pills">
    <span class="pvgt-pill">🌐 Panoramic 3D Reconstruction</span>
    <span class="pvgt-pill">📸 Multi-Frame Dense Depth</span>
    <span class="pvgt-pill">🎯 Camera Pose Estimation</span>
    <span class="pvgt-pill">☁️ Point Cloud Export</span>
    <span class="pvgt-pill">🗂️ COLMAP Compatible</span>
  </div>
  <div class="pvgt-desc">
    <p><strong>PanoVGGT</strong> is a panoramic 3D reconstruction framework built on a permutation-equivariant Vision-GNN-Transformer backbone. Given a set of panoramic images or video frames, the model jointly predicts dense depth maps, 3D point clouds, and per-frame camera poses in a single feed-forward pass — no iterative optimisation required.</p>
    <p>This demo accepts video (auto frame extraction) or a batch of images. Outputs include an interactive 3D scene viewer, per-frame radial depth maps, PLY point clouds with camera frustums, and COLMAP-format camera parameters.</p>
  </div>
  <div class="pvgt-steps">
    <div class="pvgt-step"><div class="pvgt-step-num">1</div><div class="pvgt-step-title">Upload Media</div><div class="pvgt-step-desc">Upload a panoramic video or a set of panoramic images</div></div>
    <div class="pvgt-step"><div class="pvgt-step-num">2</div><div class="pvgt-step-title">Reconstruct</div><div class="pvgt-step-desc">Click the button — PanoVGGT runs inference automatically</div></div>
    <div class="pvgt-step"><div class="pvgt-step-num">3</div><div class="pvgt-step-title">Explore</div><div class="pvgt-step-desc">Rotate, zoom and filter frames in the 3D viewer</div></div>
    <div class="pvgt-step"><div class="pvgt-step-num">4</div><div class="pvgt-step-title">Export</div><div class="pvgt-step-desc">Download PLY point clouds or COLMAP camera files</div></div>
  </div>
  <hr class="pvgt-divider">
</div>
"""

_FOOTER_HTML = """
<style>
  .pvgt-footer{text-align:center;padding:18px 20px 24px;font-family:'Space Grotesk',sans-serif;font-size:0.80rem;color:rgba(167,243,208,0.50)!important;border-top:1px solid rgba(52,211,153,0.13);margin-top:36px;}
  .pvgt-footer a{color:rgba(52,211,153,0.70)!important;text-decoration:none;transition:color 0.2s;}
  .pvgt-footer a:hover{color:#34d399!important;}
  .pvgt-footer .sep{margin:0 8px;opacity:0.4;}
</style>
<div class="pvgt-footer">
  <strong style="color:rgba(52,211,153,0.65)!important">PanoVGGT</strong>
  <span class="sep">·</span> Panoramic Visual Geometry Grounded Transformer
  <span class="sep">·</span> Built with <a href="https://www.gradio.app" target="_blank">Gradio</a>
</div>
"""


def _sec(icon: str, title: str) -> str:
    return f'<p class="section-title">{icon}&nbsp;&nbsp;{title}</p>'


def _build_scene_covers_html(scene_names: List[str]) -> str:
    """Render a cover-image card grid for all discovered example scenes."""
    abs_example = os.path.abspath(_EXAMPLE_DIR)
    cards = []
    for name in scene_names:
        img_dir = _scene_image_dir(name)
        imgs = sorted(
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        if not imgs:
            continue
        # absolute path → Gradio serves via /file=<abs_path>
        cover_abs = os.path.abspath(os.path.join(img_dir, imgs[0]))
        cards.append(
            f'<div class="scene-card">'
            f'  <img src="/file={cover_abs}" alt="{name}" '
            f'       onerror="this.style.display=\'none\'">'
            f'  <div class="scene-card-label">{name}</div>'
            f'</div>'
        )
    return f'<div class="scene-grid">{"".join(cards)}</div>'


# =========================================================================
#  5.  Gradio UI
# =========================================================================

def build_ui() -> gr.Blocks:
    theme = gr.themes.Ocean()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    # ── discover scenes at startup ────────────────────────────────────────
    scene_names = _discover_scenes()
    print(f"[PanoVGGT] Found {len(scene_names)} example scene(s): {scene_names}")

    # gr.Examples rows: [scene_name, show_cam, colormap, log_scale, ply, cam_scale]
    examples_table = [
        [name, True, "turbo", True, True, 0.05]
        for name in scene_names
    ]

    with gr.Blocks(theme=theme, css=_CSS, title="PanoVGGT Demo") as demo:

        is_example        = gr.Textbox(visible=False, value="None")
        target_dir_output = gr.Textbox(visible=False, value="None")

        gr.HTML(_HEADER_HTML)

        with gr.Row(equal_height=False):

            # ── LEFT column ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                with gr.Group():
                    gr.HTML(_sec("📂", "Upload Media"))
                    input_video = gr.Video(
                        label="Panoramic Video (mp4 / mov / …)",
                        interactive=True)
                    input_images = gr.File(
                        file_count="multiple",
                        label="Or: Panoramic Image Set (jpg / png / …)",
                        interactive=True)
                    interval = gr.Number(
                        value=None, label="Sampling Interval",
                        info=("Video: extract every N-th frame.  "
                              "Images: keep every N-th image (1 = keep all)."),
                        precision=0)

                with gr.Group():
                    gr.HTML(_sec("🎨", "Depth Visualisation Options"))
                    depth_colormap = gr.Dropdown(
                        choices=["turbo","viridis","plasma","magma",
                                 "inferno","jet","rainbow","coolwarm"],
                        value="turbo", label="Pseudo-colour Colormap")
                    use_log_scale = gr.Checkbox(
                        label="Logarithmic Depth Scale (recommended for large scenes)",
                        value=True)

                with gr.Group():
                    gr.HTML(_sec("☁️", "Point Cloud Export Options"))
                    save_ply = gr.Checkbox(
                        label="Export PLY Point Cloud + Camera Frustums", value=True)
                    camera_scale = gr.Slider(
                        minimum=0.01, maximum=0.30, value=0.05, step=0.01,
                        label="Camera Frustum Scale",
                        info="Controls the size of camera frustums in the PLY export.")

                with gr.Group():
                    gr.HTML(_sec("🖼️", "Input Frame Preview"))
                    image_gallery = gr.Gallery(
                        label="", columns=4, height="260px",
                        show_download_button=True,
                        object_fit="contain", preview=True)

            # ── RIGHT column ──────────────────────────────────────────────
            with gr.Column(scale=2):

                with gr.Group():
                    gr.HTML(_sec("🚀", "3D Reconstruction"))
                    with gr.Row():
                        submit_btn = gr.Button(
                            "▶  Run Reconstruction", scale=3, variant="primary")
                        clear_btn = gr.ClearButton(value="🗑  Clear", scale=1)
                    log_output = gr.Markdown(
                        value="Upload media and press **Run Reconstruction** to begin.",
                        elem_classes=["status-box"])

                with gr.Group():
                    gr.HTML(_sec("🌐", "Interactive 3D Scene Viewer"))
                    reconstruction_output = gr.Model3D(
                        height=500, zoom_speed=0.5, pan_speed=0.5,
                        label="PanoVGGT Output  "
                              "(drag to rotate · scroll to zoom · right-drag to pan)")

                with gr.Group():
                    gr.HTML(_sec("🎛️", "Visualisation Controls"))
                    with gr.Row():
                        show_cam = gr.Checkbox(
                            label="Show Camera Trajectory", value=True)
                        frame_filter = gr.Dropdown(
                            choices=["All"], value="All",
                            label="Point Cloud: Show Frame",
                            info='"All" displays points from every frame.')

                with gr.Group():
                    gr.HTML(_sec("📊", "Per-Frame Radial Depth Maps"))
                    depth_gallery = gr.Gallery(
                        label="Radial Depth — one image per frame",
                        columns=4, height="220px",
                        show_download_button=True,
                        object_fit="contain", preview=True)
                    depth_comparison = gr.Image(
                        label="RGB vs. Depth Comparison Grid",
                        show_download_button=True)

                with gr.Group():
                    gr.HTML(_sec("💾", "Point Cloud Download"))
                    ply_download = gr.File(
                        label="Download Main Point Cloud (pointcloud.ply)",
                        interactive=False)
                    gr.Markdown(
                        "**Exported files**\n\n"
                        "| File | Description |\n"
                        "|------|-------------|\n"
                        "| `pointcloud.ply` | Dense point cloud |\n"
                        "| `camera_frustums.ply` | Camera frustum geometry |\n"
                        "| `pointcloud_with_cameras.ply` | Combined scene |\n"
                        "| `colmap_format/` | COLMAP-compatible camera parameters |\n"
                        "| `visualize.py` | Standalone visualisation script |\n\n"
                        "```bash\n"
                        "# Generate a paper-quality figure:\n"
                        "python visualize.py --screenshot figure.png\n"
                        "```")

        # ── Example Scenes ────────────────────────────────────────────────
        if examples_table:
            with gr.Group():
                gr.HTML(_sec("🎬", "Example Scenes"))
                gr.Markdown(
                    f"**{len(scene_names)} built-in scene(s) available.** "
                    "Click any row in the table to run reconstruction automatically.")

                # ── cover image grid (visual index) ───────────────────────
                gr.HTML(_build_scene_covers_html(scene_names))

                # ── hidden textbox: receives scene_name from gr.Examples ──
                scene_selector = gr.Textbox(visible=False, label="Scene Name")

                # ── Examples table ────────────────────────────────────────
                # Each row = one scene.
                # Clicking a row fills scene_selector then calls example_pipeline.
                gr.Examples(
                    examples=examples_table,
                    inputs=[
                        scene_selector,   # ← scene name (hidden)
                        show_cam,
                        depth_colormap,
                        use_log_scale,
                        save_ply,
                        camera_scale,
                    ],
                    outputs=[
                        reconstruction_output,
                        log_output,
                        target_dir_output,
                        frame_filter,
                        image_gallery,
                        depth_gallery,
                        depth_comparison,
                        ply_download,
                    ],
                    fn=example_pipeline,
                    cache_examples=False,
                    run_on_click=True,
                    label="Click any row to reconstruct ↓",
                )

        gr.HTML(_FOOTER_HTML)

        # ── Clear ─────────────────────────────────────────────────────────
        clear_btn.add([
            input_video, input_images,
            reconstruction_output, log_output,
            target_dir_output, image_gallery,
            interval, depth_gallery,
            depth_comparison, ply_download,
        ])

        # ── Event wiring ──────────────────────────────────────────────────
        for ctrl in (input_video, input_images):
            ctrl.change(
                fn=update_gallery_on_upload,
                inputs=[input_video, input_images, interval],
                outputs=[reconstruction_output, target_dir_output,
                         image_gallery, log_output])

        def _run(td, ff, sc, dc, ls, sp, cs):
            return gradio_reconstruct(td, ff, sc, dc, ls, sp, cs)

        (submit_btn
            .click(lambda: None, outputs=[reconstruction_output])
            .then(lambda: "⏳ PanoVGGT is running inference — please wait …",
                  outputs=[log_output])
            .then(_run,
                  inputs=[target_dir_output, frame_filter, show_cam,
                          depth_colormap, use_log_scale, save_ply, camera_scale],
                  outputs=[reconstruction_output, log_output, frame_filter,
                           depth_gallery, depth_comparison, ply_download])
            .then(lambda: "False", outputs=[is_example]))

        for ctrl in (frame_filter, show_cam):
            ctrl.change(
                fn=update_visualization,
                inputs=[target_dir_output, frame_filter, show_cam, is_example],
                outputs=[reconstruction_output, log_output])

    return demo


# =========================================================================
#  6.  Entry Point
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PanoVGGT — Panoramic Visual Geometry Grounded Transformer · Gradio Demo")
    p.add_argument("--config",     type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device",     type=str, default="cuda", choices=["cuda","cpu"])
    p.add_argument("--port",       type=int, default=7860)
    p.add_argument("--share",      action="store_true")
    p.add_argument("--tmp-dir",    type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.tmp_dir:
        os.makedirs(args.tmp_dir, exist_ok=True)
        os.environ["GRADIO_TEMP_DIR"] = args.tmp_dir
        os.environ["TMPDIR"]          = args.tmp_dir

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")

    print("=" * 62)
    print("  PanoVGGT · Panoramic Visual Geometry Grounded Transformer")
    print("=" * 62)
    model = load_model(args.config, args.checkpoint, args.device)
    model.eval().to(args.device)
    print(f"[PanoVGGT] Model ready on {args.device}")
    print("=" * 62)

    demo = build_ui()
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        show_error=True,
        allowed_paths=[os.path.abspath(_EXAMPLE_DIR)],  # serve cover images
    )
#!/usr/bin/env python3
"""
Evaluate a panoramic 3D reconstruction model across all supported datasets.

Computes three families of metrics per dataset:
  • **Pose**  — pairwise rotation / translation AUC
  • **Depth** — AbsRel, RMSE, δ-thresholds, …
  • **Point cloud** — accuracy, completion, normal consistency (multi-view fusion)

Both ``world_points`` (= local_points × predicted_pose) and ``global_points``
(= direct network prediction) are evaluated when available.

Usage
-----
    python -m evaluation.eval_allpano \\
        --ckpt  path/to/checkpoint.pt \\
        --json_root results/

    # quick smoke-test on 10 sequences per dataset
    python -m evaluation.eval_allpano \\
        --num_seqs_panocity 10 \\
        --num_seqs_matterport 10 \\
        --json_root results/quick
"""

import os
import json
import random
import argparse
import importlib
import traceback
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# dataset loaders
from training.data.datasets.panocity import PanoCityDataset
from training.data.datasets.matterport3d import Matterport3DDataset
from training.data.datasets.stanford2d3ds import Stanford2D3DSDataset
from training.data.datasets.structured3d import Structured3DDataset

# evaluation utilities
from evaluation.utils.depth import depth_evaluation
from evaluation.utils.pointcloud import eval_pointcloud
from evaluation.utils.pose import (
    se3_to_relative_pose_error,
    calculate_auc_np,
)
from evaluation.utils.geometry import closed_form_inverse_se3


# =========================================================================
#  Per-dataset thresholds for ICP / normal estimation
# =========================================================================

POINTCLOUD_CONFIG = {
    # outdoor — large scale
    "PanoCity":     {"icp_threshold": 1.0, "normal_radius": 2.0},
    # indoor — room scale
    "Matterport3D": {"icp_threshold": 0.1, "normal_radius": 0.2},
    "Stanford2D3DS":{"icp_threshold": 0.1, "normal_radius": 0.2},
    "Structured3D": {"icp_threshold": 0.1, "normal_radius": 0.2},
}

_DEFAULT_PC = {"icp_threshold": 0.1, "normal_radius": 0.2}


# =========================================================================
#  Helpers
# =========================================================================

def set_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_se3_4x4(mat: torch.Tensor) -> torch.Tensor:
    """Pad ``(N, 3, 4)`` → ``(N, 4, 4)`` by appending ``[0 0 0 1]``."""
    if mat.shape[-2:] == (4, 4):
        return mat
    N = mat.shape[0]
    row = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
    return torch.cat([mat, row.view(1, 1, 4).expand(N, -1, -1)], dim=1)


def _safe_mean(values: list) -> float:
    return float(np.mean(values)) if values else 0.0


def _ensure_nhw3(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape a point-map tensor to ``(N, H, W, 3)``."""
    if tensor.dim() == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() == 4 and tensor.shape[1] == 3:
        tensor = tensor.permute(0, 2, 3, 1)
    return tensor


# =========================================================================
#  Single-sequence evaluation
# =========================================================================

def evaluate_sequence(
    model: torch.nn.Module,
    data: Dict[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    depth_max: float = 10.0,
    depth_align: str = "median-scale",
    dataset_name: str = "",
) -> tuple:
    """Run inference on one sequence and compute all metrics.

    Returns:
        ``(pose_metrics, depth_metrics, world_point_metrics, global_point_metrics)``
    """
    # ── unpack ground truth ──────────────────────────────────────────
    images = torch.stack(data["images"], dim=0).to(device)       # (N, 3, H, W)
    N, _, H, W = images.shape

    gt_extri = torch.from_numpy(np.stack(data["extrinsics"], 0)).to(device)   # (N, 3, 4) w2c
    gt_world = torch.stack(data["world_points"], 0).permute(0, 3, 1, 2).to(device)  # (N, 3, H, W)
    gt_depth = torch.stack(data["depths"], 0).to(device)                              # (N, H, W)
    gt_mask = torch.stack(data["point_masks"], 0).bool().to(device)                   # (N, H, W)

    # ── forward pass ─────────────────────────────────────────────────
    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=(amp_dtype is not None), dtype=amp_dtype or torch.float32
    ):
        pred = model(images=images)

    # ── parse predictions ────────────────────────────────────────────
    def _squeeze_batch(t):
        return t[0] if (t is not None and t.dim() >= 4 and t.shape[0] == 1) else t

    pred_extri = _squeeze_batch(pred.get("camera_poses"))          # (N, 3|4, 4) c2w
    pred_depth = _squeeze_batch(pred.get("depth"))                 # (N, H, W[, 1])
    pred_world = _squeeze_batch(pred.get("world_points"))          # (N, H, W, 3)
    pred_global = _squeeze_batch(pred.get("global_points"))        # (N, H, W, 3)

    if pred_depth is not None:
        if pred_depth.dim() == 4 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)
        if pred_depth.shape[-2:] != (H, W):
            pred_depth = torch.nn.functional.interpolate(
                pred_depth[:, None], size=(H, W), mode="bilinear", align_corners=False
            )[:, 0]
        pred_depth = pred_depth.clamp_min(0)

    if pred_world is not None:
        pred_world = _ensure_nhw3(pred_world)
    if pred_global is not None:
        pred_global = _ensure_nhw3(pred_global)

    # ── 1. Pose metrics ──────────────────────────────────────────────
    pose_metrics: dict = {}
    if pred_extri is not None and pred_extri.shape[0] == N:
        pred_se3 = _to_se3_4x4(pred_extri)
        gt_se3 = _to_se3_4x4(gt_extri)
        pred_w2c = closed_form_inverse_se3(pred_se3)

        r_err, t_err = se3_to_relative_pose_error(pred_w2c, gt_se3, N)

        if r_err.numel() > 0:
            rn, tn = r_err.cpu().numpy(), t_err.cpu().numpy()
            pose_metrics = {
                "AUC@30": calculate_auc_np(rn, tn, 30),
                "AUC@15": calculate_auc_np(rn, tn, 15),
                "AUC@5":  calculate_auc_np(rn, tn, 5),
                "AUC@3":  calculate_auc_np(rn, tn, 3),
                "R_mean": float(r_err.mean()), "R_med": float(r_err.median()),
                "T_mean": float(t_err.mean()), "T_med": float(t_err.median()),
            }
        else:
            pose_metrics = {k: 0.0 for k in
                           ["AUC@30", "AUC@15", "AUC@5", "AUC@3",
                            "R_mean", "R_med", "T_mean", "T_med"]}

    # ── 2. Depth metrics ─────────────────────────────────────────────
    depth_metrics: dict = {}
    if pred_depth is not None:
        per_frame: Dict[str, list] = {}
        align_kw = {"align_with_lad2": True} if depth_align == "scale&shift" else {"align_with_scale": True}
        for i in range(N):
            res, *_ = depth_evaluation(
                pred_depth[i].cpu().numpy(),
                gt_depth[i].cpu().numpy(),
                max_depth=depth_max,
                custom_mask=gt_mask[i].cpu().numpy(),
                **align_kw,
            )
            for k, v in res.items():
                per_frame.setdefault(k, []).append(v)
        depth_metrics = {k: _safe_mean(v) for k, v in per_frame.items()}

    # ── 3. Point-cloud metrics ───────────────────────────────────────
    pc_cfg = POINTCLOUD_CONFIG.get(dataset_name, _DEFAULT_PC)
    gt_world_np = gt_world.cpu().float().numpy().transpose(0, 2, 3, 1)  # (N,H,W,3)
    mask_np = gt_mask.cpu().numpy()

    def _eval_points(pred_pts_tensor):
        if pred_pts_tensor is None:
            return {}
        pred_np = pred_pts_tensor.cpu().float().numpy()
        all_pred, all_gt = [], []
        for i in range(N):
            m = mask_np[i]
            all_pred.append(pred_np[i][m])
            all_gt.append(gt_world_np[i][m])
        p = np.concatenate(all_pred) if all_pred else np.empty((0, 3))
        g = np.concatenate(all_gt) if all_gt else np.empty((0, 3))
        return eval_pointcloud(p, g, **pc_cfg)

    world_pt_metrics = _eval_points(pred_world)
    global_pt_metrics = _eval_points(pred_global)

    return pose_metrics, depth_metrics, world_pt_metrics, global_pt_metrics


# =========================================================================
#  Dataset-level loop
# =========================================================================

def eval_one_dataset(
    name: str,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    amp_dtype,
    num_seqs: int,
    num_frames: int,
    depth_align: str,
    json_root: str,
) -> dict:
    """Evaluate *model* on *dataset* and save per-dataset JSON."""
    pc_cfg = POINTCLOUD_CONFIG.get(name, _DEFAULT_PC)
    print(f"\n{'=' * 72}")
    print(f"  Dataset : {name}")
    print(f"  ICP th  : {pc_cfg['icp_threshold']}m  |  normal radius : {pc_cfg['normal_radius']}m")
    print(f"{'=' * 72}")

    indices = list(range(dataset.sequence_list_len))
    random.shuffle(indices)
    if 0 < num_seqs < len(indices):
        indices = indices[:num_seqs]

    agg = {"pose": {}, "depth": {}, "world_point": {}, "global_point": {}}

    for si in tqdm(indices, desc=name):
        try:
            data = dataset.get_data(seq_index=si, img_per_seq=num_frames, aspect_ratio=1.0)
            pose_m, depth_m, wpt_m, gpt_m = evaluate_sequence(
                model, data, device, amp_dtype,
                depth_max=dataset.depth_max,
                depth_align=depth_align,
                dataset_name=name,
            )
            for src, bucket in [(pose_m, "pose"), (depth_m, "depth"),
                                (wpt_m, "world_point"), (gpt_m, "global_point")]:
                for k, v in src.items():
                    agg[bucket].setdefault(k, []).append(v)

        except Exception as exc:
            print(f"\n  [ERROR] seq {si}: {exc}")
            traceback.print_exc()

    result = {cat: {k: _safe_mean(v) for k, v in metrics.items()}
              for cat, metrics in agg.items()}

    # print summary
    for cat in ("pose", "depth", "world_point", "global_point"):
        if result[cat]:
            rounded = {k: round(v, 5) for k, v in result[cat].items()}
            print(f"  {cat:>14s}: {rounded}")

    # save
    os.makedirs(json_root, exist_ok=True)
    path = os.path.join(json_root, f"eval_{name.lower()}.json")
    with open(path, "w") as f:
        json.dump({**result, "num_sequences": len(indices),
                   "frames_per_seq": num_frames,
                   "pointcloud_config": pc_cfg}, f, indent=2)
    print(f"  → {path}")
    return result


# =========================================================================
#  CLI
# =========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate on all panoramic datasets (pose / depth / point-cloud).")

    # paths
    p.add_argument("--panocity_root",       type=str, required=True)
    p.add_argument("--matterport_root",   type=str, required=True)
    p.add_argument("--stanford_root",     type=str, required=True)
    p.add_argument("--structured3d_root", type=str, required=True)

    # model
    p.add_argument("--ckpt",  type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--model", type=str, default="panovggt.models.panovggt_model:PanoVGGTModel",
                   help="module:ClassName")
    p.add_argument("--model_kwargs", type=str, default=None,
                   help="JSON string of model constructor kwargs")
    p.add_argument("--split", type=str, default="test_final",
                   choices=["train", "test", "test_final"])

    # per-dataset sequence counts (-1 = all)
    p.add_argument("--num_seqs_panocity",       type=int, default=-1)
    p.add_argument("--num_seqs_matterport",   type=int, default=-1)
    p.add_argument("--num_seqs_stanford",     type=int, default=-1)
    p.add_argument("--num_seqs_structured3d", type=int, default=100)

    # per-dataset frame counts
    p.add_argument("--frames_panocity",       type=int, default=10)
    p.add_argument("--frames_matterport",   type=int, default=3)
    p.add_argument("--frames_stanford",     type=int, default=3)
    p.add_argument("--frames_structured3d", type=int, default=3)

    # evaluation
    p.add_argument("--device",      type=str, default="cuda")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--amp_dtype",   type=str, default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--depth_align", type=str, default="median-scale")
    p.add_argument("--json_root",   type=str, default="eval_results")

    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp_dtype)

    # ── load model ───────────────────────────────────────────────────
    mod_path, cls_name = args.model.split(":")
    ModelClass = getattr(importlib.import_module(mod_path), cls_name)
    kwargs = json.loads(args.model_kwargs) if args.model_kwargs else {}
    model = ModelClass(**kwargs).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"Model loaded from {args.ckpt}")

    # ── common dataset config ────────────────────────────────────────
    common = SimpleNamespace(
        img_size=336, patch_size=14, rescale=None, rescale_aug=None,
        landscape_check=False, training=False, get_nearby=True,
        inside_random=False, allow_duplicate_img=True, augs=None, debug=False,
    )

    all_results = {}

    # ── datasets ─────────────────────────────────────────────────────
    DATASETS = [
        ("PanoCity", PanoCityDataset, {
            "PanoCity_DIR": args.panocity_root,
            "min_num_images": max(2, args.frames_panocity),
        }, args.num_seqs_panocity, args.frames_panocity),

        ("Matterport3D", Matterport3DDataset, {
            "Matterport3D_DIR": args.matterport_root,
            "min_num_images": max(2, args.frames_matterport),
        }, args.num_seqs_matterport, args.frames_matterport),

        ("Stanford2D3DS", Stanford2D3DSDataset, {
            "Stanford2D3DS_DIR": args.stanford_root,
            "min_num_images": max(2, args.frames_stanford),
        }, args.num_seqs_stanford, args.frames_stanford),

        ("Structured3D", Structured3DDataset, {
            "Structured3D_DIR": args.structured3d_root,
            "min_num_rooms": max(2, args.frames_structured3d),
        }, args.num_seqs_structured3d, args.frames_structured3d),
    ]

    for name, DatasetClass, ds_kwargs, n_seqs, n_frames in DATASETS:
        ds = DatasetClass(
            common_conf=common, split=args.split,
            len_train=10**9, len_test=10**9,
            expand_ratio=3, augmentation=None,
            **ds_kwargs,
        )
        all_results[name] = eval_one_dataset(
            name, ds, model, device, amp_dtype,
            num_seqs=n_seqs, num_frames=n_frames,
            depth_align=args.depth_align,
            json_root=args.json_root,
        )

    # ── global summary ───────────────────────────────────────────────
    summary_path = os.path.join(args.json_root, "eval_all_datasets.json")
    with open(summary_path, "w") as f:
        json.dump({"results": all_results, "config": vars(args)}, f, indent=2)

    print(f"\n{'=' * 72}")
    print(f"  All done — summary saved to {summary_path}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
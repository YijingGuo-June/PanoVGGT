"""
Monocular / panoramic depth evaluation.

Supports several alignment strategies (median-scale, least-squares,
scale-and-shift, L1 robust) and computes the standard suite of metrics
(AbsRel, SqRel, RMSE, log-RMSE, δ-thresholds, …).
"""

import numpy as np
import torch

__all__ = ["depth_evaluation"]


# ── private alignment helpers ──────────────────────────────────────────


def _absolute_value_scaling(
    pred: torch.Tensor,
    gt: torch.Tensor,
    s: torch.Tensor,
    lr: float = 1e-1,
    max_iters: int = 100,
) -> tuple:
    """L1-optimal scale & shift via iteratively-reweighted LS (Weiszfeld)."""
    t = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    for _ in range(max_iters):
        residuals = (s * pred + t - gt).abs().clamp(min=1e-8)
        w = 1.0 / residuals
        sw = (w * pred).sum()
        s = (w * pred * gt).sum() / sw.clamp(min=1e-12)
        t = (w * (gt - s * pred)).sum() / w.sum().clamp(min=1e-12)
    return s.detach(), t.detach()


def _absolute_value_scaling2(
    pred: torch.Tensor,
    gt: torch.Tensor,
    s_init: float = 1.0,
    lr: float = 1e-4,
    max_iters: int = 1000,
) -> tuple:
    """L1-optimal scale & shift via SGD (useful when Weiszfeld is unstable)."""
    s = torch.tensor(s_init, device=pred.device, dtype=pred.dtype, requires_grad=True)
    t = torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
    optim = torch.optim.Adam([s, t], lr=lr)
    for _ in range(max_iters):
        optim.zero_grad()
        loss = (s * pred + t - gt).abs().mean()
        loss.backward()
        optim.step()
    return s.detach(), t.detach()


def _depth_to_disparity(depth: torch.Tensor) -> torch.Tensor:
    return 1.0 / depth.clamp(min=1e-8)


# ── main API ───────────────────────────────────────────────────────────


def depth_evaluation(
    predicted_depth,
    ground_truth_depth,
    max_depth: float = 80.0,
    custom_mask=None,
    # clipping
    pre_clip_min: float = None,
    pre_clip_max: float = None,
    post_clip_min: float = None,
    post_clip_max: float = None,
    # alignment modes (mutually exclusive — first match wins)
    metric_scale: bool = False,
    align_with_lstsq: bool = False,
    align_with_lad: bool = False,
    align_with_lad2: bool = False,
    align_with_scale: bool = False,
    disp_input: bool = False,
    # L1-alignment hyper-params
    lr: float = 1e-4,
    max_iters: int = 1000,
    use_gpu: bool = False,
):
    """Evaluate a predicted depth map against ground truth.

    Args:
        predicted_depth: ``(H, W)`` or ``(N, H, W)`` predicted depth.
        ground_truth_depth: Same shape as *predicted_depth*.
        max_depth: Ignore GT pixels beyond this depth.
        custom_mask: Optional boolean validity mask (same spatial shape).
        metric_scale: If ``True``, skip alignment entirely.
        align_with_lstsq: Least-squares scale + shift.
        align_with_lad: Weiszfeld L1 scale + shift.
        align_with_lad2: SGD-based L1 scale + shift.
        align_with_scale: Robust scale-only (no shift).
        disp_input: Operate in disparity space.

    Returns:
        ``(results_dict, error_map, aligned_pred, masked_gt)``
    """
    # ── tensor conversion ────────────────────────────────────────────
    if isinstance(predicted_depth, np.ndarray):
        predicted_depth = torch.from_numpy(predicted_depth)
    if isinstance(ground_truth_depth, np.ndarray):
        ground_truth_depth = torch.from_numpy(ground_truth_depth)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    predicted_depth_original = predicted_depth.clone()
    ground_truth_depth_original = ground_truth_depth.clone()

    # flatten (N,H,W) → (N*H, W) so everything is 2-D
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.reshape(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.reshape(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.reshape(-1, w)

    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()

    # ── valid-pixel mask ─────────────────────────────────────────────
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = ground_truth_depth_original > 0

    pred = predicted_depth_original[mask]
    gt = ground_truth_depth_original[mask]

    if pre_clip_min is not None:
        pred = pred.clamp(min=pre_clip_min)
    if pre_clip_max is not None:
        pred = pred.clamp(max=pre_clip_max)

    if disp_input:
        real_gt = gt.clone()
        gt = 1.0 / (gt + 1e-8)

    # ── alignment ────────────────────────────────────────────────────
    s = t = scale_factor = None

    if metric_scale:
        pass  # no alignment
    elif align_with_lstsq:
        A = np.hstack([pred.cpu().numpy().reshape(-1, 1),
                        np.ones((pred.numel(), 1))])
        sol = np.linalg.lstsq(A, gt.cpu().numpy().reshape(-1, 1), rcond=None)[0]
        s = torch.tensor(sol[0, 0], device=pred.device)
        t = torch.tensor(sol[1, 0], device=pred.device)
        pred = s * pred + t
    elif align_with_lad:
        s_init = torch.median(gt) / torch.median(pred)
        s, t = _absolute_value_scaling(pred, gt, s_init)
        pred = s * pred + t
    elif align_with_lad2:
        s_init = (torch.median(gt) / torch.median(pred)).item()
        s, t = _absolute_value_scaling2(pred, gt, s_init=s_init, lr=lr, max_iters=max_iters)
        pred = s * pred + t
    elif align_with_scale:
        s = torch.nanmean(gt) / torch.nanmean(pred)
        for _ in range(10):
            residuals = (s * pred - gt).abs().clamp(min=1e-8)
            w = 1.0 / residuals
            s = (w * pred * gt).sum() / (w * pred ** 2).sum()
        s = s.clamp(min=1e-3).detach()
        pred = s * pred
    else:
        # default: median-scale
        scale_factor = torch.median(gt) / torch.median(pred)
        pred = pred * scale_factor

    if disp_input:
        gt = real_gt
        pred = _depth_to_disparity(pred)

    if post_clip_min is not None:
        pred = pred.clamp(min=post_clip_min)
    if post_clip_max is not None:
        pred = pred.clamp(max=post_clip_max)

    # ── apply custom mask inside the valid region ────────────────────
    if custom_mask is not None:
        inner = custom_mask[mask]
        pred = pred[inner]
        gt = gt[inner]

    # ── metrics ──────────────────────────────────────────────────────
    if pred.numel() == 0:
        empty = {
            "mae": 0, "mse": 0, "rmse": 0,
            "Abs Rel": 0, "Sq Rel": 0, "log10": 0, "rmse_log": 0,
            "δ < 1.25": 0, "δ < 1.25^2": 0, "δ < 1.25^3": 0,
            "valid_pixels": 0,
        }
        z = torch.zeros_like(ground_truth_depth_original)
        return empty, z, predicted_depth_original, z

    diff = pred - gt
    abs_diff = diff.abs()

    abs_rel = (abs_diff / gt).mean().item()
    sq_rel = ((diff ** 2) / gt).mean().item()
    mae = abs_diff.mean().item()
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5
    log10 = (torch.log10(pred.clamp(min=1e-6)) - torch.log10(gt.clamp(min=1e-6))).abs().mean().item()
    log_rmse = ((torch.log(pred.clamp(min=1e-5)) - torch.log(gt)) ** 2).mean().sqrt().item()

    ratio = torch.maximum(pred / gt, gt / pred)
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()

    num_valid = pred.numel() if custom_mask is None else int(inner.sum().item())

    results = {
        "mae": mae, "mse": mse, "rmse": rmse,
        "Abs Rel": abs_rel, "Sq Rel": sq_rel, "log10": log10, "rmse_log": log_rmse,
        "δ < 1.25": d1, "δ < 1.25^2": d2, "δ < 1.25^3": d3,
        "valid_pixels": num_valid,
    }

    # ── per-pixel error map (for visualisation) ──────────────────────
    if s is not None and t is not None:
        aligned_full = s * predicted_depth_original + t
    elif s is not None:
        aligned_full = s * predicted_depth_original
    elif scale_factor is not None:
        aligned_full = scale_factor * predicted_depth_original
    else:
        aligned_full = predicted_depth_original

    if disp_input:
        aligned_full = _depth_to_disparity(aligned_full)

    err_map = (aligned_full - ground_truth_depth_original).abs() / ground_truth_depth_original.clamp(min=1e-8)
    err_map = torch.where(mask, err_map, torch.zeros_like(err_map))
    gt_map = torch.where(mask, ground_truth_depth_original, torch.zeros_like(ground_truth_depth_original))

    return results, err_map, aligned_full, gt_map
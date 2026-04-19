"""
Relative camera-pose evaluation metrics.

Computes pairwise rotation and translation angular errors, and the
standard AUC@τ aggregation used in visual localisation benchmarks.

Reference: https://github.com/facebookresearch/vggsfm
"""

import numpy as np
import torch

from evaluation.utils.geometry import mat_to_quat, closed_form_inverse_se3

__all__ = [
    "se3_to_relative_pose_error",
    "rotation_angle",
    "translation_angle",
    "calculate_auc",
    "calculate_auc_np",
]


# =========================================================================
#  Pair indexing
# =========================================================================

def _build_pair_index(N: int, B: int = 1):
    """Return two flat index tensors ``(i1, i2)`` for all unordered pairs."""
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1 = (i1_[None] + torch.arange(B)[:, None] * N).reshape(-1)
    i2 = (i2_[None] + torch.arange(B)[:, None] * N).reshape(-1)
    return i1, i2


# =========================================================================
#  Rotation & translation angular errors
# =========================================================================

def rotation_angle(
    rot_gt: torch.Tensor,
    rot_pred: torch.Tensor,
    eps: float = 1e-15,
) -> torch.Tensor:
    """Geodesic angular error (degrees) between rotation matrices.

    Args:
        rot_gt:   ``(P, 3, 3)`` ground-truth rotations.
        rot_pred: ``(P, 3, 3)`` predicted rotations.

    Returns:
        ``(P,)`` error in degrees.
    """
    q_gt = mat_to_quat(rot_gt)
    q_pred = mat_to_quat(rot_pred)

    dot = (q_pred * q_gt).sum(dim=1)
    loss = (1.0 - dot ** 2).clamp(min=eps)
    return torch.arccos(1.0 - 2.0 * loss) * (180.0 / np.pi)


def translation_angle(
    t_gt: torch.Tensor,
    t_pred: torch.Tensor,
    eps: float = 1e-15,
    default_err: float = 1e6,
    resolve_sign_ambiguity: bool = True,
) -> torch.Tensor:
    """Angular error (degrees) between translation directions.

    Args:
        t_gt:   ``(P, 3)`` ground-truth translations.
        t_pred: ``(P, 3)`` predicted translations.
        resolve_sign_ambiguity: Take ``min(err, 180° − err)``.

    Returns:
        ``(P,)`` error in degrees.
    """
    t_gt = t_gt / (t_gt.norm(dim=1, keepdim=True) + eps)
    t_pred = t_pred / (t_pred.norm(dim=1, keepdim=True) + eps)

    dot2 = (t_gt * t_pred).sum(dim=1) ** 2
    err = torch.arccos(torch.sqrt(1.0 - (1.0 - dot2).clamp(min=eps)))
    err[torch.isnan(err) | torch.isinf(err)] = default_err

    deg = err * (180.0 / np.pi)
    if resolve_sign_ambiguity:
        deg = torch.min(deg, (180.0 - deg).abs())
    return deg


# =========================================================================
#  AUC computation
# =========================================================================

def calculate_auc(
    r_error: torch.Tensor,
    t_error: torch.Tensor,
    max_threshold: int = 30,
) -> float:
    """Area Under the pose-error Curve (torch version).

    ``err = max(R_err, T_err)`` is binned into ``[0, max_threshold]``
    and the normalised CDF is averaged.
    """
    errs = torch.maximum(r_error, t_error)
    hist = torch.histc(errs.float(), bins=max_threshold + 1, min=0, max=max_threshold)
    return float(torch.cumsum(hist / max(errs.numel(), 1), dim=0).mean())


def calculate_auc_np(
    r_error: np.ndarray,
    t_error: np.ndarray,
    max_threshold: int = 30,
) -> float:
    """Area Under the pose-error Curve (numpy version)."""
    errs = np.maximum(r_error, t_error)
    hist, _ = np.histogram(errs, bins=np.arange(max_threshold + 1))
    norm = hist.astype(float) / max(len(errs), 1)
    return float(np.mean(np.cumsum(norm)))


# =========================================================================
#  End-to-end: SE(3) → relative-pose errors
# =========================================================================

def se3_to_relative_pose_error(
    pred_se3: torch.Tensor,
    gt_se3: torch.Tensor,
    num_frames: int,
    trans_norm_thresh: float = 1e-2,
):
    """Compute pairwise relative-pose errors, skipping near-zero baselines.

    Args:
        pred_se3: ``(N, 4, 4)`` predicted poses (c2w **or** w2c — must match *gt*).
        gt_se3:   ``(N, 4, 4)`` ground-truth poses.
        num_frames: Number of frames ``N``.
        trans_norm_thresh: Skip pairs whose GT translation norm is below this.

    Returns:
        ``(r_error, t_error)`` — rotation (°) and translation (°) errors for
        valid pairs.  Both may be empty if all pairs are below threshold.
    """
    i1, i2 = _build_pair_index(num_frames)

    rel_gt = gt_se3[i2].bmm(closed_form_inverse_se3(gt_se3[i1]))
    rel_pred = pred_se3[i2].bmm(closed_form_inverse_se3(pred_se3[i1]))

    r_err = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])

    t_gt = rel_gt[:, :3, 3]
    valid = t_gt.norm(p=2, dim=-1) > trans_norm_thresh

    if valid.sum() == 0:
        empty = torch.empty(0, dtype=r_err.dtype, device=r_err.device)
        return empty, empty.clone()

    t_err = translation_angle(t_gt[valid], rel_pred[:, :3, 3][valid])
    return r_err[valid], t_err
"""
Multi-view point-cloud evaluation metrics.

Includes Umeyama Sim(3) alignment, accuracy / completion / normal
consistency, IoU, and a combined ICP-based evaluation pipeline.

Reference: https://github.com/CUT3R/CUT3R/blob/main/eval/mv_recon/utils.py
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree

__all__ = [
    "umeyama",
    "accuracy",
    "completion",
    "completion_ratio",
    "compute_iou",
    "eval_pointcloud",
]


# =========================================================================
#  Umeyama Sim(3) alignment
# =========================================================================

def umeyama(X: np.ndarray, Y: np.ndarray):
    """Estimate the Sim(3) transform ``c * R @ X + t ≈ Y``.

    Args:
        X: ``(d, n)`` source points.
        Y: ``(d, n)`` target points (must share the same indexing).

    Returns:
        ``(c, R, t)`` — scale ``float``, rotation ``(d, d)``, translation ``(d, 1)``.
    """
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ Vt
    t = mu_y - c * R @ mu_x
    return c, R, t


# =========================================================================
#  Per-direction metrics
# =========================================================================

def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Reconstruction → GT distances (how close predicted points are to GT).

    Returns:
        ``(mean, median, nc_mean, nc_median, mae, rmse)``
        Normal-consistency values are ``None`` when normals are not provided.
    """
    tree = KDTree(gt_points)
    dists, idx = tree.query(rec_points, workers=-1)

    acc_mean = float(np.mean(dists))
    acc_med = float(np.median(dists))
    mae = acc_mean
    rmse = float(np.sqrt(np.mean(dists ** 2)))

    if gt_normals is not None and rec_normals is not None:
        dots = np.abs(np.sum(gt_normals[idx] * rec_normals, axis=-1))
        return acc_mean, acc_med, float(np.mean(dots)), float(np.median(dots)), mae, rmse

    return acc_mean, acc_med, None, None, mae, rmse


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """GT → reconstruction distances (how much of the GT is covered).

    Returns:
        ``(mean, median, nc_mean, nc_median)``
    """
    tree = KDTree(rec_points)
    dists, idx = tree.query(gt_points, workers=-1)

    comp_mean = float(np.mean(dists))
    comp_med = float(np.median(dists))

    if gt_normals is not None and rec_normals is not None:
        dots = np.abs(np.sum(gt_normals * rec_normals[idx], axis=-1))
        return comp_mean, comp_med, float(np.mean(dots)), float(np.median(dots))

    return comp_mean, comp_med, None, None


def completion_ratio(gt_points, rec_points, dist_th: float = 0.05) -> float:
    """Fraction of GT points within *dist_th* of a reconstructed point."""
    tree = KDTree(rec_points)
    dists, _ = tree.query(gt_points)
    return float(np.mean((dists < dist_th).astype(np.float32)))


def compute_iou(pred_vox, target_vox) -> float:
    """Voxel-grid IoU between two Open3D ``VoxelGrid`` objects."""
    s1 = {tuple(np.round(v.grid_index, 4)) for v in pred_vox.get_voxels()}
    s2 = {tuple(np.round(v.grid_index, 4)) for v in target_vox.get_voxels()}
    return len(s1 & s2) / max(len(s1 | s2), 1)


# =========================================================================
#  Combined evaluation pipeline (Umeyama + ICP + metrics)
# =========================================================================

_EMPTY_METRICS = {
    "Acc-mean": 0.0, "Acc-med": 0.0,
    "Comp-mean": 0.0, "Comp-med": 0.0,
    "overall_mean": 0.0, "overall_med": 0.0,
    "NC1-mean": 0.0, "NC1-med": 0.0,
    "NC2-mean": 0.0, "NC2-med": 0.0,
    "MAE": 0.0, "RMSE": 0.0,
}


def eval_pointcloud(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    icp_threshold: float = 0.1,
    normal_radius: float = 0.2,
) -> dict:
    """Full evaluation of a merged multi-view point cloud.

    1. Umeyama coarse alignment (Sim(3))
    2. ICP fine alignment
    3. Accuracy / completion / normal consistency

    Args:
        pred_pts: ``(M, 3)`` predicted world points.
        gt_pts:   ``(K, 3)`` ground-truth world points.
        icp_threshold: Max correspondence distance for ICP (metres).
        normal_radius: Search radius for normal estimation (metres).

    Returns:
        Dict of metric names → float values.
    """
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return dict(_EMPTY_METRICS)

    # 1. Umeyama coarse alignment
    c, R, t = umeyama(pred_pts.T, gt_pts.T)
    pred_aligned = c * (R @ pred_pts.T).T + t.T

    # 2. Build Open3D clouds
    pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_aligned))
    pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))

    # 3. ICP refinement
    reg = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_gt, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    pcd_pred.transform(reg.transformation)

    # 4. Normal estimation
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    pcd_pred.estimate_normals(search_param=search)
    pcd_gt.estimate_normals(search_param=search)

    pred_n = np.asarray(pcd_pred.normals)
    gt_n = np.asarray(pcd_gt.normals)
    pred_p = np.asarray(pcd_pred.points)
    gt_p = np.asarray(pcd_gt.points)

    # 5. Metrics
    acc, acc_med, nc1, nc1_med, mae, rmse = accuracy(gt_p, pred_p, gt_n, pred_n)
    comp, comp_med, nc2, nc2_med = completion(gt_p, pred_p, gt_n, pred_n)

    return {
        "Acc-mean": acc, "Acc-med": acc_med,
        "Comp-mean": comp, "Comp-med": comp_med,
        "overall_mean": (acc + comp) / 2,
        "overall_med": (acc_med + comp_med) / 2,
        "NC1-mean": nc1 or 0.0, "NC1-med": nc1_med or 0.0,
        "NC2-mean": nc2 or 0.0, "NC2-med": nc2_med or 0.0,
        "MAE": mae, "RMSE": rmse,
    }
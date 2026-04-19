"""
Low-level SE(3) and quaternion helpers used across all evaluation code.
"""

import numpy as np
import torch


def mat_to_quat(rot: torch.Tensor) -> torch.Tensor:
    """Convert a batch of 3×3 rotation matrices to unit quaternions (w, x, y, z).

    Uses Shepperd's method for numerical stability.

    Args:
        rot: ``(N, 3, 3)`` rotation matrices.

    Returns:
        ``(N, 4)`` unit quaternions ordered ``(w, x, y, z)``.
    """
    batch = rot.shape[0]
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]

    quat = torch.zeros(batch, 4, dtype=rot.dtype, device=rot.device)

    s = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2  # 4w
    quat[:, 0] = 0.25 * s
    quat[:, 1] = (rot[:, 2, 1] - rot[:, 1, 2]) / s
    quat[:, 2] = (rot[:, 0, 2] - rot[:, 2, 0]) / s
    quat[:, 3] = (rot[:, 1, 0] - rot[:, 0, 1]) / s

    # Fix cases where trace is small — use the largest diagonal entry instead
    for i in range(batch):
        t = trace[i]
        if t > 0:
            continue
        R = rot[i]
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s_ = 2.0 * torch.sqrt((1.0 + R[0, 0] - R[1, 1] - R[2, 2]).clamp(min=1e-10))
            quat[i, 0] = (R[2, 1] - R[1, 2]) / s_
            quat[i, 1] = 0.25 * s_
            quat[i, 2] = (R[0, 1] + R[1, 0]) / s_
            quat[i, 3] = (R[0, 2] + R[2, 0]) / s_
        elif R[1, 1] > R[2, 2]:
            s_ = 2.0 * torch.sqrt((1.0 + R[1, 1] - R[0, 0] - R[2, 2]).clamp(min=1e-10))
            quat[i, 0] = (R[0, 2] - R[2, 0]) / s_
            quat[i, 1] = (R[0, 1] + R[1, 0]) / s_
            quat[i, 2] = 0.25 * s_
            quat[i, 3] = (R[1, 2] + R[2, 1]) / s_
        else:
            s_ = 2.0 * torch.sqrt((1.0 + R[2, 2] - R[0, 0] - R[1, 1]).clamp(min=1e-10))
            quat[i, 0] = (R[1, 0] - R[0, 1]) / s_
            quat[i, 1] = (R[0, 2] + R[2, 0]) / s_
            quat[i, 2] = (R[1, 2] + R[2, 1]) / s_
            quat[i, 3] = 0.25 * s_

    return quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-10)


def closed_form_inverse_se3(
    se3: torch.Tensor,
    R: torch.Tensor = None,
    T: torch.Tensor = None,
) -> torch.Tensor:
    """Batch-invert SE(3) matrices using the closed-form R^T / −R^T t.

    Args:
        se3: ``(N, 4, 4)`` or ``(N, 3, 4)`` rigid transforms.
        R:   Optional pre-extracted ``(N, 3, 3)`` rotation.
        T:   Optional pre-extracted ``(N, 3, 1)`` translation.

    Returns:
        ``(N, 4, 4)`` inverted transforms.
    """
    if se3.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"se3 must be (N,4,4) or (N,3,4), got {se3.shape}")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    R_t = R.transpose(1, 2)
    t_inv = -torch.bmm(R_t, T)

    inv = torch.eye(4, dtype=se3.dtype, device=se3.device).unsqueeze(0).expand(len(R), -1, -1).clone()
    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = t_inv
    return inv


def closed_form_inverse_se3_np(se3: np.ndarray) -> np.ndarray:
    """Numpy version of :func:`closed_form_inverse_se3`."""
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    R_t = np.transpose(R, (0, 2, 1))
    t_inv = -np.matmul(R_t, T)
    inv = np.tile(np.eye(4), (len(R), 1, 1))
    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = t_inv
    return inv
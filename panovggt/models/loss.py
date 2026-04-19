"""
Loss Functions for 3D Vision Tasks

This module provides loss functions for joint camera pose estimation and 3D point prediction.
It includes scale-invariant point losses, normal consistency losses, and relative pose losses.
"""

import math
import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from panovggt.utils.geometry import homogenize_points, se3_inverse, depth_edge
from panovggt.utils.alignment import align_points_scale


logger = logging.getLogger(__name__)


# =============================================================================
# Utility Functions
# =============================================================================

def weighted_mean(
    x: torch.Tensor,
    w: Optional[torch.Tensor] = None,
    dim: Optional[Union[int, torch.Size]] = None,
    keepdim: bool = False,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Compute weighted mean along specified dimensions.
    
    Args:
        x (torch.Tensor): Input tensor.
        w (torch.Tensor, optional): Weight tensor with same shape as x.
        dim (int or torch.Size, optional): Dimension(s) to reduce.
        keepdim (bool): Whether to keep reduced dimensions.
        eps (float): Small constant for numerical stability.
    
    Returns:
        torch.Tensor: Weighted mean of x.
    """
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def smooth_l1(
    err: torch.Tensor,
    beta: float = 0.0
) -> torch.Tensor:
    """
    Smooth L1 loss (Huber loss variant).
    
    Args:
        err (torch.Tensor): Error tensor.
        beta (float): Threshold for switching between L1 and L2. If 0, returns err.
    
    Returns:
        torch.Tensor: Smoothed error.
    """
    if beta == 0:
        return err
    else:
        return torch.where(
            err < beta,
            0.5 * err.square() / beta,
            err - 0.5 * beta
        )


def angle_diff_vec3(
    v1: torch.Tensor,
    v2: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute angular difference between two 3D vectors using atan2 for numerical stability.
    
    Args:
        v1 (torch.Tensor): First vector of shape [..., 3].
        v2 (torch.Tensor): Second vector of shape [..., 3].
        eps (float): Small constant for numerical stability.
    
    Returns:
        torch.Tensor: Angular difference in radians, shape [...], range [0, π].
    """
    # Compute cross product norm (proportional to sin(theta))
    cross_product = torch.cross(v1, v2, dim=-1)
    cross_norm = torch.norm(cross_product, dim=-1)
    
    # Compute dot product (proportional to cos(theta))
    dot_product = (v1 * v2).sum(dim=-1)
    
    # Use atan2 for numerically stable angle computation
    angle = torch.atan2(cross_norm + eps, dot_product)
    
    # Ensure angle is in [0, π]
    angle = torch.abs(angle)
    angle = torch.clamp(angle, min=0.0, max=math.pi)
    
    return angle


# =============================================================================
# Point Loss
# =============================================================================

class PointLoss(nn.Module):
    """
    Scale-Invariant Point Loss with Normal Consistency.
    
    This loss computes:
    1. L1 loss on scale-aligned 3D points (local and optional global).
    2. Normal consistency loss based on cross products of neighboring points.
    3. Optional confidence loss for uncertainty estimation.
    
    Args:
        local_align_res (int): Number of points to sample for scale alignment. Default: 4096.
        train_conf (bool): Whether to train confidence prediction. Default: False.
        expected_dist_thresh (float): Distance threshold for positive confidence samples. Default: 0.02.
    """
    
    def __init__(
        self,
        local_align_res: int = 4096,
        train_conf: bool = False,
        expected_dist_thresh: float = 0.02
    ):
        super().__init__()
        self.local_align_res = local_align_res
        self.train_conf = train_conf
        self.expected_dist_thresh = expected_dist_thresh
        
        # Loss functions
        self.criteria_local = nn.L1Loss(reduction='none')
        if self.train_conf:
            self.conf_loss_fn = nn.BCEWithLogitsLoss()
    
    def prepare_sampling(
        self,
        pts: torch.Tensor,
        mask: torch.Tensor,
        target_size: int = 4096
    ) -> torch.Tensor:
        """
        Sample fixed number of valid points for robust scale estimation.
        
        Args:
            pts (torch.Tensor): Points of shape [B, N, H, W, C].
            mask (torch.Tensor): Valid mask of shape [B, N, H, W].
            target_size (int): Number of points to sample.
        
        Returns:
            torch.Tensor: Sampled points of shape [B, target_size, C].
        """
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]  # [M, C]
            
            if valid_pts.shape[0] > 0:
                # Resample to target size
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # [1, C, M]
                valid_pts = F.interpolate(
                    valid_pts, size=target_size, mode='nearest'
                )  # [1, C, target_size]
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # [target_size, C]
            else:
                # Fallback to ones if no valid points
                valid_pts = torch.ones(
                    (target_size, C),
                    device=pts.device,
                    dtype=pts.dtype
                )
            
            output.append(valid_pts)
        
        return torch.stack(output, dim=0)
    
    def compute_normal_loss(
        self,
        points: torch.Tensor,
        gt_points: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normal consistency loss using cross products of neighboring points.
        
        Args:
            points (torch.Tensor): Predicted points of shape [B, N, H, W, 3].
            gt_points (torch.Tensor): Ground truth points of shape [B, N, H, W, 3].
            mask (torch.Tensor): Valid mask of shape [B, N, H, W].
        
        Returns:
            torch.Tensor: Scalar normal loss.
        """
        # Detect edges using radial distance
        radial_distance = torch.norm(gt_points, dim=-1)
        not_edge = ~depth_edge(radial_distance, rtol=0.03)
        mask = torch.logical_and(mask, not_edge)
        
        # Extract 4 corner points for each pixel
        leftup = points[..., :-1, :-1, :]
        rightup = points[..., :-1, 1:, :]
        leftdown = points[..., 1:, :-1, :]
        rightdown = points[..., 1:, 1:, :]
        
        # Compute normals via cross products (4 triangles per pixel)
        upxleft = torch.cross(
            rightup - rightdown, leftdown - rightdown, dim=-1
        )
        leftxdown = torch.cross(
            leftup - rightup, rightdown - rightup, dim=-1
        )
        downxright = torch.cross(
            leftdown - leftup, rightup - leftup, dim=-1
        )
        rightxup = torch.cross(
            rightdown - leftdown, leftup - leftdown, dim=-1
        )
        
        # Same for ground truth
        gt_leftup = gt_points[..., :-1, :-1, :]
        gt_rightup = gt_points[..., :-1, 1:, :]
        gt_leftdown = gt_points[..., 1:, :-1, :]
        gt_rightdown = gt_points[..., 1:, 1:, :]
        
        gt_upxleft = torch.cross(
            gt_rightup - gt_rightdown, gt_leftdown - gt_rightdown, dim=-1
        )
        gt_leftxdown = torch.cross(
            gt_leftup - gt_rightup, gt_rightdown - gt_rightup, dim=-1
        )
        gt_downxright = torch.cross(
            gt_leftdown - gt_leftup, gt_rightup - gt_leftup, dim=-1
        )
        gt_rightxup = torch.cross(
            gt_rightdown - gt_leftdown, gt_leftup - gt_leftdown, dim=-1
        )
        
        # Compute validity masks for each triangle
        mask_leftup = mask[..., :-1, :-1]
        mask_rightup = mask[..., :-1, 1:]
        mask_leftdown = mask[..., 1:, :-1]
        mask_rightdown = mask[..., 1:, 1:]
        
        mask_upxleft = mask_rightup & mask_leftdown & mask_rightdown
        mask_leftxdown = mask_leftup & mask_rightdown & mask_rightup
        mask_downxright = mask_leftdown & mask_rightup & mask_leftup
        mask_rightxup = mask_rightdown & mask_leftup & mask_leftdown
        
        # Compute angular differences with smoothing
        MIN_ANGLE = math.radians(1)
        MAX_ANGLE = math.radians(90)
        BETA_RAD = math.radians(3)
        
        loss = (
            mask_upxleft * smooth_l1(
                angle_diff_vec3(upxleft, gt_upxleft).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            ) +
            mask_leftxdown * smooth_l1(
                angle_diff_vec3(leftxdown, gt_leftxdown).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            ) +
            mask_downxright * smooth_l1(
                angle_diff_vec3(downxright, gt_downxright).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            ) +
            mask_rightxup * smooth_l1(
                angle_diff_vec3(rightxup, gt_rightxup).clamp(MIN_ANGLE, MAX_ANGLE),
                beta=BETA_RAD
            )
        )
        
        # Average over all pixels and triangles
        loss = loss.mean() / (4 * max(points.shape[-3:-1]))
        
        # Safety check for NaN/Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            loss = torch.tensor(0.0, device=points.device, dtype=points.dtype)
        
        return loss
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute point loss.
        
        Args:
            pred (dict): Predictions containing:
                - local_points: [B, N, H, W, 3]
                - conf (optional): [B, N, H, W, 1] or [B, N, H, W]
                - global_points (optional): [B, N, H, W, 3]
            gt (dict): Ground truth containing:
                - local_points: [B, N, H, W, 3]
                - global_points: [B, N, H, W, 3]
                - valid_masks: [B, N, H, W]
        
        Returns:
            Tuple containing:
                - total_loss (torch.Tensor): Scalar total loss.
                - details (dict): Dictionary with individual loss components.
                - scale (torch.Tensor): Estimated scale factors of shape [B].
        """
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        
        B, N, H, W, _ = pred_local_pts.shape
        
        # Compute depth weights (inverse radial distance)
        weights = torch.norm(gt_local_pts, dim=-1).sqrt()
        weights = weights.clamp_min(
            0.1 * weighted_mean(weights, valid_masks, dim=(-2, -1), keepdim=True)
        )
        weights = 1.0 / (weights + 1e-6)
        
        # Scale alignment using sampled points
        with torch.no_grad():
            xyz_pred_sampled = self.prepare_sampling(
                pred_local_pts, valid_masks, target_size=self.local_align_res
            )
            xyz_gt_sampled = self.prepare_sampling(
                gt_local_pts, valid_masks, target_size=self.local_align_res
            )
            xyz_weights_sampled = self.prepare_sampling(
                weights[..., None], valid_masks, target_size=self.local_align_res
            )[..., 0]
            
            scale = align_points_scale(
                xyz_pred_sampled, xyz_gt_sampled, xyz_weights_sampled
            )
            scale = scale.abs().clamp_min(1e-6)
        
        # Apply scale alignment
        aligned_local_pts = scale.view(B, 1, 1, 1, 1) * pred_local_pts
        
        # Compute L1 point loss with depth weighting
        local_pts_loss = self.criteria_local(
            aligned_local_pts[valid_masks].float(),
            gt_local_pts[valid_masks].float()
        ) * weights[valid_masks].float()[..., None]
        
        details = {}
        total_loss = local_pts_loss.mean()
        details['local_pts_loss'] = local_pts_loss.mean()
        
        # Confidence loss (optional)
        if self.train_conf:
            pred_conf = pred['conf']
            if pred_conf.dim() == 5 and pred_conf.shape[-1] == 1:
                pred_conf = pred_conf[..., 0]
            
            # Positive samples: error < threshold
            valid_samples = (
                local_pts_loss.detach().mean(-1) < self.expected_dist_thresh
            )
            
            conf_logits = pred_conf[valid_masks]
            local_conf_loss = self.conf_loss_fn(conf_logits, valid_samples.float())
            
            total_loss += 0.05 * local_conf_loss
            details['local_conf_loss'] = local_conf_loss
        
        # Normal consistency loss
        normal_loss = self.compute_normal_loss(
            aligned_local_pts, gt_local_pts, valid_masks
        )
        total_loss += normal_loss
        details['normal_loss'] = normal_loss
        
        # Global points loss (optional)
        if 'global_points' in pred and pred['global_points'] is not None:
            gt_global_pts = gt['global_points']
            pred_global_pts = pred['global_points'] * scale.view(B, 1, 1, 1, 1)
            
            global_pts_loss = self.criteria_local(
                pred_global_pts[valid_masks].float(),
                gt_global_pts[valid_masks].float()
            ) * weights[valid_masks].float()[..., None]
            
            total_loss += global_pts_loss.mean()
            details['global_pts_loss'] = global_pts_loss.mean()
        
        return total_loss, details, scale


# =============================================================================
# Camera Pose Loss
# =============================================================================

class CameraLoss(nn.Module):
    """
    Relative Camera Pose Loss with Adaptive Huber Loss.
    
    Computes relative pose errors between all pairs of frames using:
    1. Rotation angular error (geodesic distance on SO(3)).
    2. Translation Huber loss with adaptive threshold.
    
    Args:
        alpha (float): Weight of translation loss relative to rotation loss. Default: 100.
        delta_real_world (float): Huber loss threshold in real-world units (meters). Default: 0.25.
    """
    
    def __init__(
        self,
        alpha: float = 100.0,
        delta_real_world: float = 0.25
    ):
        super().__init__()
        self.alpha = alpha
        self.delta_real_world = delta_real_world
    
    @staticmethod
    def rotation_angular_error(
        R: torch.Tensor,
        R_gt: torch.Tensor,
        eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute rotation angular error in radians.
        
        Args:
            R (torch.Tensor): Predicted rotation matrices of shape [B, 3, 3].
            R_gt (torch.Tensor): Ground truth rotation matrices of shape [B, 3, 3].
            eps (float): Small constant for numerical stability.
        
        Returns:
            torch.Tensor: Scalar angular error in radians.
        """
        # Compute relative rotation: R_rel = R^T @ R_gt
        R_rel = torch.matmul(R.transpose(1, 2), R_gt)
        
        # Extract rotation angle from trace: cos(θ) = (trace(R_rel) - 1) / 2
        trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1.0) / 2.0
        
        # Clamp to valid range for numerical stability
        cosine = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
        
        # Compute angle in [0, π]
        angle = torch.acos(cosine)
        
        return angle.mean()
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor],
        scale: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute camera pose loss.
        
        Args:
            pred (dict): Predictions containing:
                - camera_poses: [B, N, 4, 4] in c2w format.
            gt (dict): Ground truth containing:
                - camera_poses: [B, N, 4, 4] in c2w format.
                - norm_factors: [B] normalization factors.
            scale (torch.Tensor): Scale factors from point alignment of shape [B].
        
        Returns:
            Tuple containing:
                - total_loss (torch.Tensor): Scalar total loss.
                - details (dict): Dictionary with 'trans_loss' and 'rot_loss'.
        """
        pred_pose = pred['camera_poses']  # [B, N, 4, 4] c2w
        gt_pose = gt['camera_poses']      # [B, N, 4, 4] c2w
        norm_factors = gt['norm_factors'] # [B]
        
        B, N, _, _ = pred_pose.shape
        
        # Safety check for invalid scale
        if torch.isnan(scale).any() or torch.isinf(scale).any():
            logger.error("Invalid scale in CameraLoss; using fallback value 1.0")
            scale = torch.ones_like(scale)
        
        # Apply scale alignment to predicted translations
        pred_pose_aligned = pred_pose.clone()
        pred_pose_aligned[..., :3, 3] *= scale.view(B, 1, 1)
        
        # Convert to w2c format
        pred_w2c = se3_inverse(pred_pose_aligned)
        gt_w2c = se3_inverse(gt_pose)
        
        # Safety check for NaN in w2c matrices
        if torch.isnan(pred_w2c).any() or torch.isnan(gt_w2c).any():
            logger.error("NaN detected in w2c matrices; returning zero loss")
            zero = torch.zeros((), device=pred_pose.device, dtype=pred_pose.dtype)
            return zero, {'trans_loss': zero, 'rot_loss': zero}
        
        # Compute relative poses: T_j_i = T_w_i @ T_j_w
        pred_rel_all = torch.matmul(
            pred_w2c.unsqueeze(2), pred_pose_aligned.unsqueeze(1)
        )  # [B, N, N, 4, 4]
        gt_rel_all = torch.matmul(
            gt_w2c.unsqueeze(2), gt_pose.unsqueeze(1)
        )  # [B, N, N, 4, 4]
        
        # Exclude diagonal (i == j)
        mask = ~torch.eye(N, dtype=torch.bool, device=pred_pose.device)
        
        # Extract translations and rotations
        t_pred = pred_rel_all[..., :3, 3][:, mask]   # [B, N*(N-1), 3]
        R_pred = pred_rel_all[..., :3, :3][:, mask]  # [B, N*(N-1), 3, 3]
        t_gt = gt_rel_all[..., :3, 3][:, mask]
        R_gt = gt_rel_all[..., :3, :3][:, mask]
        
        # Adaptive Huber loss for translation
        delta_normalized = self.delta_real_world / (norm_factors + 1e-8)  # [B]
        delta_expanded = delta_normalized.view(B, 1, 1)  # [B, 1, 1]
        
        error = t_pred - t_gt  # [B, P, 3] where P = N*(N-1)
        abs_error = torch.abs(error)
        
        quadratic = 0.5 * (error ** 2)
        linear = delta_expanded * abs_error - 0.5 * (delta_expanded ** 2)
        
        loss_per_element = torch.where(
            abs_error <= delta_expanded, quadratic, linear
        )
        trans_loss = loss_per_element.mean()
        
        # Rotation angular error
        rot_loss = self.rotation_angular_error(
            R_pred.reshape(-1, 3, 3),
            R_gt.reshape(-1, 3, 3)
        )
        
        # Total loss
        total_loss = self.alpha * trans_loss + rot_loss
        
        # Safety check for NaN/Inf
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            logger.error("NaN/Inf detected in total camera loss; returning zero")
            zero = torch.zeros((), device=pred_pose.device, dtype=pred_pose.dtype)
            return zero, {'trans_loss': zero, 'rot_loss': zero}
        
        return total_loss, {'trans_loss': trans_loss, 'rot_loss': rot_loss}


# =============================================================================
# Combined Loss
# =============================================================================

class Loss(nn.Module):
    """
    Combined Loss for Joint Point and Camera Estimation.
    
    This module combines:
    1. Point loss (local/global points + normal consistency + optional confidence).
    2. Camera pose loss (relative rotation + translation).
    
    Args:
        train_conf (bool): Whether to train confidence prediction. Default: False.
    """
    
    def __init__(self, train_conf: bool = False):
        super().__init__()
        self.point_loss = PointLoss(train_conf=train_conf)
        self.camera_loss = CameraLoss()
    
    def prepare_gt(self, gt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare ground truth data for loss computation.
        
        Converts extrinsics from w2c to c2w format and ensures 4x4 matrices.
        
        Args:
            gt (dict): Raw ground truth containing:
                - extrinsics: [B, N, 3, 4] or [B, N, 4, 4] in w2c format.
                - world_points: [B, N, H, W, 3].
                - cam_points: [B, N, H, W, 3].
                - point_masks: [B, N, H, W].
                - images: [B, N, C, H, W].
                - norm_factors (optional): [B].
                - depths (optional): [B, N, H, W].
        
        Returns:
            dict: Processed ground truth with c2w camera poses.
        """
        poses_w2c = gt['extrinsics']
        
        # Convert [B, N, 3, 4] to [B, N, 4, 4] if needed
        if poses_w2c.shape[-2:] == (3, 4):
            B, N, _, _ = poses_w2c.shape
            bottom = poses_w2c.new_zeros((B, N, 1, 4))
            bottom[..., 0, 3] = 1.0
            poses_w2c = torch.cat([poses_w2c, bottom], dim=-2)
        
        # Convert w2c to c2w
        poses_c2w = se3_inverse(poses_w2c)
        
        return {
            'imgs': gt['images'],
            'global_points': gt['world_points'],
            'local_points': gt['cam_points'],
            'valid_masks': gt['point_masks'],
            'camera_poses': poses_c2w,
            'depths': gt.get('depths', None),
            'norm_factors': gt.get('norm_factors', None),
        }
    
    def normalize_pred(
        self,
        pred: Dict[str, torch.Tensor],
        gt: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Normalize predictions using scale computed from local points.
        
        Args:
            pred (dict): Predictions containing:
                - local_points: [B, N, H, W, 3].
                - camera_poses: [B, N, 4, 4] in c2w format.
                - global_points (optional): [B, N, H, W, 3].
            gt (dict): Ground truth with 'valid_masks'.
        
        Returns:
            dict: Normalized predictions.
        """
        local_points = pred['local_points']
        camera_poses = pred['camera_poses']
        masks = gt['valid_masks']
        
        B, N, H, W, _ = local_points.shape
        
        # Compute normalization scale from local points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        denom = masks.float().sum(dim=[-1, -2, -3]).clamp(min=1e-8)
        norm_factor = all_dis.sum(dim=[-1, -2]) / denom  # [B]
        
        scale = norm_factor.view(B, 1, 1, 1, 1)
        
        # Normalize local points
        pred['local_points'] = local_points / scale
        
        # Normalize global points if present
        if 'global_points' in pred and pred['global_points'] is not None:
            global_points = pred['global_points']
            
            # Transform to first camera coordinate system
            R0 = camera_poses[:, 0, :3, :3]  # [B, 3, 3]
            t0 = camera_poses[:, 0, :3, 3]   # [B, 3]
            
            # Compute w2c translation: t_w2c = -(t_c2w @ R_c2w)
            t_w2c = -torch.matmul(t0.unsqueeze(-2), R0).squeeze(-2)  # [B, 3]
            
            # Transform: x_cam0 = x_world @ R0 + t_w2c
            global_points_cam0 = torch.matmul(
                global_points,
                R0.unsqueeze(1).unsqueeze(2)
            ) + t_w2c.view(B, 1, 1, 1, 3)
            
            pred['global_points'] = global_points_cam0 / scale
        
        # Normalize camera translations
        camera_poses_normalized = camera_poses.clone()
        camera_poses_normalized[..., :3, 3] /= norm_factor.view(B, 1, 1)
        pred['camera_poses'] = camera_poses_normalized
        
        return pred
    
    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_raw: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss and individual components.
        
        Args:
            pred (dict): Model predictions.
            gt_raw (dict): Raw ground truth data.
        
        Returns:
            dict: Loss dictionary containing:
                - loss_objective: Total loss for optimization.
                - loss_camera: Camera pose loss.
                - loss_T: Translation loss.
                - loss_R: Rotation loss.
                - loss_conf_point: Point confidence loss.
                - loss_reg_point: Point regression loss.
                - loss_grad_point: Normal consistency loss.
                - loss_local_point: Local point loss.
                - loss_global_point: Global point loss.
                - loss_conf_depth: Depth confidence loss (unused, set to 0).
                - loss_reg_depth: Depth regression loss (unused, set to 0).
                - loss_grad_depth: Depth gradient loss (unused, set to 0).
        """
        # Prepare ground truth
        gt = self.prepare_gt(gt_raw)
        
        # Normalize predictions
        pred = self.normalize_pred(pred, gt)
        
        # Compute point and camera losses
        point_loss, point_details, scale = self.point_loss(pred, gt)
        cam_loss, cam_details = self.camera_loss(pred, gt, scale)
        
        # Total objective loss
        loss_objective = point_loss + 0.1 * cam_loss
        
        # Helper to convert to tensor
        def as_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            return torch.tensor(
                x, device=loss_objective.device, dtype=loss_objective.dtype
            )
        
        zero = loss_objective.new_tensor(0.0)
        
        # Construct unified loss dictionary
        loss_dict = {
            # Total objective
            'loss_objective': loss_objective,
            
            # Camera losses
            'loss_camera': as_tensor(cam_loss),
            'loss_T': as_tensor(cam_details.get('trans_loss', zero)),
            'loss_R': as_tensor(cam_details.get('rot_loss', zero)),
            
            # Depth losses (not currently used, set to zero)
            'loss_conf_depth': zero,
            'loss_reg_depth': zero,
            'loss_grad_depth': zero,
            
            # Point losses
            'loss_conf_point': as_tensor(point_details.get('local_conf_loss', zero)),
            'loss_reg_point': (
                as_tensor(point_details.get('local_pts_loss', zero)) +
                as_tensor(point_details.get('global_pts_loss', zero))
            ),
            'loss_grad_point': as_tensor(point_details.get('normal_loss', zero)),
            'loss_local_point': as_tensor(point_details.get('local_pts_loss', zero)),
            'loss_global_point': as_tensor(point_details.get('global_pts_loss', zero)),
        }
        
        return loss_dict

"""
Camera Pose Estimation Module

This module provides neural network components for estimating camera poses from visual features.
It includes residual convolutional blocks and a camera head that outputs SE(3) transformations.
"""

import logging
from typing import Tuple, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class ResConvBlock(nn.Module):
    """
    Residual Block with Linear Transformations.
    
    Applies three sequential linear transformations with ReLU activations
    and a skip connection for stable gradient flow.
    
    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Skip branch used when channel dimensions differ.
        self.head_skip = (
            nn.Identity() if in_channels == out_channels 
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        
        # Three-layer residual MLP path.
        self.res_conv1 = nn.Linear(in_channels, out_channels)
        self.res_conv2 = nn.Linear(out_channels, out_channels)
        self.res_conv3 = nn.Linear(out_channels, out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape [..., C_in].
            
        Returns:
            torch.Tensor: Output tensor of shape [..., C_out].
        """
        identity = self.head_skip(x)
        
        out = F.relu(self.res_conv1(x))
        out = F.relu(self.res_conv2(out))
        out = F.relu(self.res_conv3(out))
        
        return identity + out


class CameraHead(nn.Module):
    """
    Camera Pose Estimation Head.
    
    Predicts camera pose (rotation + translation) from visual features.
    Outputs SE(3) transformation matrices with numerically stable SVD-based
    rotation orthogonalization.
    
    Args:
        dim (int): Feature dimension. Default: 512.
    """
    
    def __init__(self, dim: int = 512):
        super().__init__()
        self.dim = dim
        
        # Stacked residual blocks for feature refinement.
        output_dim = dim
        self.res_conv = nn.ModuleList([
            deepcopy(ResConvBlock(output_dim, output_dim)) 
            for _ in range(2)
        ])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Extra MLP layers before pose heads.
        self.more_mlps = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        
        # Pose prediction heads
        self.fc_t = nn.Linear(dim, 3)
        self.fc_rot = nn.Linear(dim, 9)
        
    def forward(
        self, 
        feat: torch.Tensor, 
        patch_h: int, 
        patch_w: int
    ) -> torch.Tensor:
        """
        Predict camera pose from features.
        
        Args:
            feat (torch.Tensor): Input features of shape [B*N, H*W, C].
            patch_h (int): Patch height.
            patch_w (int): Patch width.
            
        Returns:
            torch.Tensor: Camera pose matrices of shape [B*N, 4, 4] in SE(3).
        """
        BN, hw, c = feat.shape
        
        # Apply residual blocks
        for i in range(2):
            feat = self.res_conv[i](feat)
        
        # Spatial pooling
        feat = self.avgpool(
            feat.permute(0, 2, 1).reshape(BN, -1, patch_h, patch_w).contiguous()
        )
        feat = feat.view(feat.size(0), -1)
        
        # Feature refinement
        feat = self.more_mlps(feat)
        
        # Predict pose in FP32 for numerical stability
        with torch.amp.autocast(device_type='cuda', enabled=False):
            feat = feat.float()
            
            out_t = self.fc_t(feat)
            out_r = self.fc_rot(feat)
            
            # Validate outputs
            if not self._is_valid_tensor(out_r):
                logger.error("Invalid rotation prediction (NaN/Inf detected)")
                
            if not self._is_valid_tensor(out_t):
                logger.error("Invalid translation prediction (NaN/Inf detected)")
            
            # Convert to 4x4 transformation matrix
            pose = self._build_pose_matrix(out_r, out_t, BN, feat.device)
        
        return pose
    
    def _build_pose_matrix(
        self,
        rotation_raw: torch.Tensor,
        translation: torch.Tensor,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Build 4x4 SE(3) pose matrix from rotation and translation.
        
        Args:
            rotation_raw (torch.Tensor): Raw rotation matrix [B, 9].
            translation (torch.Tensor): Translation vector [B, 3].
            batch_size (int): Batch size.
            device (torch.device): Target device.
            
        Returns:
            torch.Tensor: Pose matrix [B, 4, 4].
        """
        if not self._is_valid_tensor(rotation_raw):
            logger.warning("Using identity rotation due to invalid input")
            rotation = torch.eye(3, device=device, dtype=rotation_raw.dtype)
            rotation = rotation.unsqueeze(0).expand(batch_size, 3, 3).contiguous()
        else:
            rotation = self._orthogonalize_rotation(rotation_raw)
        
        if not self._is_valid_tensor(rotation):
            logger.error("Rotation orthogonalization failed, using identity")
            rotation = torch.eye(3, device=device, dtype=rotation.dtype)
            rotation = rotation.unsqueeze(0).expand(batch_size, 3, 3).contiguous()
        
        pose = torch.zeros(batch_size, 4, 4, device=device, dtype=rotation.dtype)
        pose[:, :3, :3] = rotation
        pose[:, :3, 3] = translation
        pose[:, 3, 3] = 1.0
        
        return pose
    
    def _orthogonalize_rotation(
        self, 
        matrix: torch.Tensor, 
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Orthogonalize rotation matrix using SVD projection onto SO(3).
        
        Args:
            matrix (torch.Tensor): Input matrix [B, 9] or [B, 3, 3].
            eps (float): Numerical stability threshold.
            
        Returns:
            torch.Tensor: Orthogonalized rotation matrix [B, 3, 3].
        """
        original_dtype = matrix.dtype
        matrix = matrix.float()
        
        if matrix.dim() < 3:
            matrix = matrix.reshape(-1, 3, 3)
        
        batch_size = matrix.shape[0]
        
        # Normalize rows
        row_norms = torch.norm(matrix, p=2, dim=-1, keepdim=True)
        
        zero_mask = row_norms < eps
        if zero_mask.any():
            num_zeros = zero_mask.sum().item()
            logger.warning(
                f"Found {num_zeros} near-zero rows in rotation matrix, "
                f"using unit vectors"
            )
            row_norms = torch.where(zero_mask, torch.ones_like(row_norms), row_norms)
        
        matrix_normalized = matrix / row_norms
        matrix_transposed = matrix_normalized.transpose(-2, -1)
        
        # SVD decomposition
        try:
            U, S, Vh = torch.linalg.svd(matrix_transposed)
            V = Vh.transpose(-2, -1)
            
            if not (self._is_valid_tensor(U) and self._is_valid_tensor(V)):
                raise RuntimeError("SVD produced invalid outputs (NaN/Inf)")
                
        except Exception as e:
            logger.error(f"SVD decomposition failed: {e}")
            identity = torch.eye(3, device=matrix.device, dtype=torch.float32)
            return identity.unsqueeze(0).expand(batch_size, 3, 3).to(original_dtype)
        
        # Compute rotation
        rotation = torch.matmul(V, U.transpose(-2, -1))
        
        # Ensure det(R) = +1
        det = torch.det(rotation)
        
        if not self._is_valid_tensor(det):
            logger.warning("Invalid determinant in rotation matrix")
            det = torch.ones_like(det)
        
        det_sign = torch.sign(det).view(-1, 1, 1)
        correction = torch.ones(
            batch_size, 3, 3, device=matrix.device, dtype=torch.float32
        )
        correction[:, :, 2] = det_sign.squeeze(-1)
        
        V_corrected = V * correction
        rotation = torch.matmul(V_corrected, U.transpose(-2, -1))
        
        # Final validation
        if not self._is_valid_tensor(rotation):
            logger.error("Final rotation matrix contains NaN/Inf")
            identity = torch.eye(3, device=matrix.device, dtype=torch.float32)
            return identity.unsqueeze(0).expand(batch_size, 3, 3).to(original_dtype)
        
        return rotation.to(original_dtype)
    
    @staticmethod
    def _is_valid_tensor(tensor: torch.Tensor) -> bool:
        """Check if tensor contains only finite values."""
        return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())



# Compatibility Alias (Optional)

def convert_pose_to_4x4(self, B, out_r, out_t, device):
    """Legacy method name for backward compatibility."""
    return self._build_pose_matrix(out_r, out_t, B, device)


def svd_orthogonalize(self, m, eps=1e-8):
    """Legacy method name for backward compatibility."""
    return self._orthogonalize_rotation(m, eps)


# Attach legacy methods
CameraHead.convert_pose_to_4x4 = convert_pose_to_4x4
CameraHead.svd_orthogonalize = svd_orthogonalize
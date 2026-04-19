# Equirectangular panorama rotation module.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .. import Conversion

class EquirecRotate(nn.Module):
    def __init__(self, equ_h):
        super().__init__()
        self.equ_h = equ_h
        self.equ_w = equ_h * 2
        
        # Precompute unit-sphere rays in float32 for sampling.
        X = torch.arange(self.equ_w, dtype=torch.float32)[None, :, None].repeat(self.equ_h, 1, 1)
        Y = torch.arange(self.equ_h, dtype=torch.float32)[:, None, None].repeat(1, self.equ_w, 1)
        XY = torch.cat([X, Y], dim=-1).unsqueeze(0)
        self.grid = Conversion.XY2xyz(XY, shape=(self.equ_h, self.equ_w), mode='torch') # OpenCV camera convention
    
    def forward(self, equi, axis_angle=None, rotation_matrix=None, mode='bilinear'):
        """
        Rotate an equirectangular panorama by axis-angle or rotation matrix.

        Notes:
            - Rotation may be float64 for geometry precision.
            - Sampling grid is cast to the input dtype for grid_sample.
        """
        assert mode in ['nearest', 'bilinear']
        
        # 1) Build rotation matrix.
        if axis_angle is not None:
            R = Conversion.angle_axis_to_rotation_matrix(axis_angle)
        else:
            R = rotation_matrix
            if R is None:
                raise ValueError("Either axis_angle or rotation_matrix must be provided.")

        # 2) Rotate rays and build backward-warp sampling grid.
        #    grid_sample maps output pixels to source coordinates.
        grid = self.grid.to(device=equi.device, dtype=R.dtype)
        
        # 3) Rotate xyz rays.
        xyz = (R[:, None, None, ...] @ grid[..., None]).squeeze(-1)
        
        # 4) Project rotated xyz to normalized lon/lat coordinates.
        XY = Conversion.xyz2lonlat(xyz, clip=False, mode='torch')
        X, Y = torch.unbind(XY, dim=-1)
        XY = torch.cat([(X / math.pi).unsqueeze(-1), (Y / (0.5 * math.pi)).unsqueeze(-1)], dim=-1)
        
        # 5) Sample the panorama with the computed grid.
        #    Keep grid dtype aligned with input tensor dtype.
        sample = F.grid_sample(equi, XY.to(equi.dtype), mode=mode, align_corners=True)

        return sample
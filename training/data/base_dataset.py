# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# base_dataset.py

import numpy as np
import torch
import math
from PIL import Image, ImageFile
from torchvision.transforms import functional as F

from torch.utils.data import Dataset
# Updated import to get the new functions
from .dataset_util import (
    unproject_pano_depth_to_camera_coords,
    camera_coords_to_world_coords,
    transform_pano_track_points,
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for PanoVGGTModel and VGGSfM training.
    (Docstring unchanged)
    """

    def __init__(
            self,
            common_conf,
    ):
        """
        Initialize the base dataset with common configuration.
        (Docstring unchanged)
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.
        (Docstring unchanged)
        """
        seq_index, img_per_seq, aspect_ratio = idx_N
        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.
        (Docstring unchanged)
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def process_one_image(self, image, depth_map, extrinsic_w2c, shape, equi_rotate, R_delta, depth_max, track=None):
        """
        image:  np.float32, CHW in [0,1]
        depth_map: np.float32, (1,H,W) meters
        extrinsic_w2c: np.float32, (3,4)
        shape: (H,W)
        R_delta: torch.Tensor (3,3) or None
        """
        # 1) Convert numpy inputs to CPU float tensors.
        rgb_tensor = torch.from_numpy(image).float()  # (3,H,W)
        depth_tensor = torch.from_numpy(depth_map).float()  # (1,H,W)
        # print('depth_tensor1', depth_tensor.shape)

        # 2) Convert w2c pose to c2w form.
        R_w2c = np.copy(extrinsic_w2c[:3, :3])  
        t_w2c = np.copy(extrinsic_w2c[:3, 3])  

        R_c2w = R_w2c.T
        t_c2w = -R_w2c.T @ t_w2c

        
        if (R_delta is not None) and (equi_rotate is not None):
            # dtype/device
            R_cam = R_delta.to(dtype=rgb_tensor.dtype, device=rgb_tensor.device)  # (3,3) torch

            # a) RGB / Depth
            rgb_batch = rgb_tensor.unsqueeze(0)  # (1,3,H,W)
            depth_batch = depth_tensor.unsqueeze(0)  # (1,1,H,W)

            rgb_rot = equi_rotate(rgb_batch, rotation_matrix=R_cam.unsqueeze(0), mode='bilinear')
            depth_rot = equi_rotate(depth_batch, rotation_matrix=R_cam.unsqueeze(0), mode='nearest')

            rgb_tensor = rgb_rot.squeeze(0)  # (3,H,W)
            depth_tensor = depth_rot.squeeze(0)  # (1,H,W)
            # print('depth_tensor', depth_tensor.shape)

            # b) c2w = c2w @ R_cam
            R_cam_np = R_cam.cpu().numpy()
            R_c2w_new = R_c2w @ R_cam_np  
            t_c2w_new = t_c2w  

                # c) Convert updated c2w back to w2c.
            R_w2c_new = R_c2w_new.T
            t_w2c_new = -R_w2c_new @ t_c2w_new

            # d) extrinsic (3x4)
            extrinsic_updated = np.zeros((3, 4), dtype=np.float32)
            extrinsic_updated[:3, :3] = R_w2c_new.astype(np.float32)
            extrinsic_updated[:3, 3] = t_w2c_new.astype(np.float32)

            # e) Rotate track points by inverse camera rotation (R_cam^T).
            if track is not None:
                R_inv = R_cam.transpose(0, 1).contiguous()  
                track = transform_pano_track_points(track, R_inv, shape)

            
            Rw2c_use, tw2c_use = R_w2c_new, t_w2c_new
        else:
            # No augmentation: keep original extrinsic.
            extrinsic_updated = extrinsic_w2c.astype(np.float32)
            Rw2c_use, tw2c_use = R_w2c, t_w2c

        # 4) Compute camera and world point coordinates from depth.
        cam_coords = unproject_pano_depth_to_camera_coords(depth_tensor, shape)  # (H,W,3) torch
        world_coords = camera_coords_to_world_coords(cam_coords, Rw2c_use, tw2c_use)  # (H,W,3) torch

        # 5) mask
        depth_tensor = depth_tensor.squeeze(0)  # (H,W)
        valid_mask = (depth_tensor > 0.1) & (depth_tensor <= depth_max) & (
            ~torch.isnan(depth_tensor))

        return {
            'rgb': rgb_tensor,  # (3,H,W) torch
            'depth_tensor': depth_tensor,  # (H,W) torch
            'extrinsic': extrinsic_updated,  # (3,4) np.float32
            'cam_coords': cam_coords,  # (H,W,3) torch
            'world_coords': world_coords,  # (H,W,3) torch
            'valid_mask': valid_mask,  # (H,W) torch.bool
            'track': track,
        }


    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """Sample a set of IDs from a sequence close to the median of given IDs."""
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0

        total_ids = len(ids)

        # Use median frame index as the neighborhood center.
        sorted_ids = sorted(ids)
        median_idx = sorted_ids[len(sorted_ids) // 2]  

        if expand_range is None:
            expand_range = int(total_ids * expand_ratio)

        
        low_bound = max(0, median_idx - expand_range // 2)
        high_bound = min(full_seq_num, median_idx + expand_range // 2)

        
        if low_bound == 0:
            high_bound = min(full_seq_num, expand_range)
        elif high_bound == full_seq_num:
            low_bound = max(0, full_seq_num - expand_range)

        valid_range = np.arange(low_bound, high_bound)

        # ids
        sampled_ids = np.random.choice(
            valid_range,
            size=max(0, total_ids - len(ids)),  
            replace=True,
        )

        # idsids
        result_ids = np.concatenate([ids, sampled_ids])

        return result_ids

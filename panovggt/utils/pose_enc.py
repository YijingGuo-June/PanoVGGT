# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .rotation import quat_to_mat, mat_to_quat


def extri_intri_to_pose_encoding(
    extrinsics, image_size_hw=None, pose_encoding_type="absT_quaR"  # e.g., (256, 512)
):
    """Convert camera extrinsics and intrinsics to a compact pose encoding.

    This function transforms camera parameters into a unified pose encoding format,
    which can be used for various downstream tasks like pose prediction or representation.

    Args:
        extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4,
            where B is batch size and S is sequence length.
            In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world transformation.
            The format is [R|t] where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
        image_size_hw (tuple): Tuple of (height, width) of the image in pixels.
            Required for computing field of view values. For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding to use. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).

    Returns:
        torch.Tensor: Encoded camera pose parameters with shape BxSx9.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
    """

    # extrinsics: BxSx3x4
    # intrinsics: BxSx3x3

    if pose_encoding_type == "absT_quaR":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        pose_encoding = torch.cat([T, quat], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding, pose_encoding_type="absT_quaR"
):
    """Convert a pose encoding back to camera extrinsics and intrinsics.

    This function performs the inverse operation of extri_intri_to_pose_encoding,
    reconstructing the full camera parameters from the compact encoding.

    Args:
        pose_encoding (torch.Tensor): Encoded camera pose parameters with shape BxSx9,
            where B is batch size and S is sequence length.
            For "absT_quaR_FoV" type, the 9 dimensions are:
            - [:3] = absolute translation vector T (3D)
            - [3:7] = rotation as quaternion quat (4D)
            - [7:] = field of view (2D)
            Required for reconstructing intrinsics from field of view values.
            For example: (256, 512).
        pose_encoding_type (str): Type of pose encoding used. Currently only
            supports "absT_quaR_FoV" (absolute translation, quaternion rotation, field of view).
        build_intrinsics (bool): Whether to reconstruct the intrinsics matrix.
            If False, only extrinsics are returned and intrinsics will be None.

    Returns:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): Camera extrinsic parameters with shape BxSx3x4.
              In OpenCV coordinate system (x-right, y-down, z-forward), representing camera from world
              transformation. The format is [R|t] where R is a 3x3 rotation matrix and t is
              a 3x1 translation vector.
            - intrinsics (torch.Tensor or None): Camera intrinsic parameters with shape BxSx3x3,
              or None if build_intrinsics is False. Defined in pixels, with format:
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              where fx, fy are focal lengths and (cx, cy) is the principal point,
              assumed to be at the center of the image (W/2, H/2).
    """

    if pose_encoding_type == "absT_quaR":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

    else:
        raise NotImplementedError

    return extrinsics
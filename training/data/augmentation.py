# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision.transforms import functional as F
import random
import math
import numpy as np
from typing import Optional, Dict
from torchvision import transforms
from panovggt.Projection import EquirecRotate


def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    return transforms.Compose(transform_list) if transform_list else None


class PanoAugmentation:
    """
    Applies COLOR-ONLY augmentations to a batch of panoramic images.
    Geometric augmentations like yaw rotation are handled elsewhere.
    """

    def __init__(self, aug_config: dict, training: bool):
        """
        Initializes the color augmentation pipeline with randomly sampled parameters.
        """
        self.training = training

        # --- Sample random parameters for color/gamma ONCE ---
        self.gamma = None
        self.color_jitter_params = None

        if self.training and aug_config:
            # Sample Gamma Correction
            if 'rand_gamma' in aug_config and random.random() > 0.5:
                gamma_range = aug_config['rand_gamma']
                gamma = random.uniform(gamma_range.get('min', 0.8), gamma_range.get('max', 1.2))
                self.gamma = 1.0 / gamma if random.random() < 0.5 else gamma

            # Sample Color Jitter Parameters
            if aug_config.get('color_aug', False) and random.random() > 0.5:
                b, c, s, h = 0.4, 0.4, 0.4, 0.1
                order = list(range(4))
                random.shuffle(order)
                self.color_jitter_params = {
                    'brightness_factor': random.uniform(max(0, 1 - b), 1 + b),
                    'contrast_factor': random.uniform(max(0, 1 - c), 1 + c),
                    'saturation_factor': random.uniform(max(0, 1 - s), 1 + s),
                    'hue_factor': random.uniform(-h, h),
                    'order': order
                }

    def __call__(self, images_tensor: torch.Tensor):
        """
        Applies the pre-determined color augmentations to a batch of images.

        Args:
            images_tensor (torch.Tensor): Batch of images, shape (N, C, H, W).

        Returns:
            torch.Tensor: The augmented images tensor.
        """
        if not self.training:
            return images_tensor

        # Apply Color Jitter to each image in the batch
        if self.color_jitter_params is not None:
            params = self.color_jitter_params
            for i in range(len(images_tensor)):
                img = images_tensor[i]
                for op in params['order']:
                    if op == 0:
                        img = F.adjust_brightness(img, params['brightness_factor'])
                    elif op == 1:
                        img = F.adjust_contrast(img, params['contrast_factor'])
                    elif op == 2:
                        img = F.adjust_saturation(img, params['saturation_factor'])
                    elif op == 3:
                        img = F.adjust_hue(img, params['hue_factor'])
                images_tensor[i] = img

        # Apply Gamma Correction to the whole batch
        if self.gamma is not None:
            for i in range(len(images_tensor)):
                images_tensor[i] = F.adjust_gamma(images_tensor[i], self.gamma)

        return [torch.clamp(img, 0.0, 1.0) for img in images_tensor]

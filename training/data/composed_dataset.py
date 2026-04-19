# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

from hydra.utils import instantiate
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
import bisect
from .dataset_util import *
from .track_util import *
from .augmentation import PanoAugmentation


class ComposedDataset(Dataset, ABC):
    """
    Composes multiple base datasets and applies common configurations.

    This dataset provides a flexible way to combine multiple base datasets while
    applying shared augmentations, track generation, and other processing steps.
    It handles image normalization, tensor conversion, and other preparations
    needed for training computer vision models with sequences of images.
    """
    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.augmentation_config = common_config.augs

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        # --- Track Settings ---
        # Whether to include point tracks in the output
        self.load_track = common_config.load_track
        # Number of point tracks to include per sequence
        self.track_num = common_config.track_num

        # --- Mode Settings ---
        # Whether the dataset is being used for training (affects augmentations)
        self.training = common_config.training
        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples

    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.
        It expects the base dataset to return a dictionary of Tensors.
        """
        # --- Stage 1: Get the already-processed batch of Tensors ---
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # This call directly returns a dictionary of Tensors (e.g., from PanoSUNCGDataset).
        # No more NumPy-to-Tensor conversion is needed here.
        sample = self.base_dataset[idx_tuple]

        # --- Stage 2: Apply the FINAL layer of augmentation (Color) ---
        if self.training:
            # Create a fresh PanoAugmentation instance for each batch to ensure randomness.
            pano_color_aug = PanoAugmentation(
                aug_config=self.augmentation_config,
                training=self.training
            )
            # Apply color augmentation directly to the images tensor.
            sample['images'] = pano_color_aug(sample['images'])

        # --- Stage 3: Track Processing (if enabled) ---
        if self.load_track:
            # This logic assumes the base dataset might return pre-computed tracks (as NumPy)
            # or requires on-the-fly generation from the processed Tensors.
            if "tracks" in sample and sample["tracks"] is not None:
                # If tracks are pre-computed (e.g., as NumPy), convert them now.
                if isinstance(sample["tracks"], np.ndarray):
                    tracks = torch.from_numpy(sample["tracks"].astype(np.float32))
                    track_vis_mask = torch.from_numpy(sample["track_masks"].astype(bool))
                else:  # Assume they are already Tensors
                    tracks = sample["tracks"]
                    track_vis_mask = sample["track_masks"]

                valid_indices = torch.where(track_vis_mask[0])[0]
                if len(valid_indices) >= self.track_num:
                    sampled_indices = valid_indices[torch.randperm(len(valid_indices))][:self.track_num]
                else:
                    sampled_indices = valid_indices[torch.randint(0, len(valid_indices), (self.track_num,))]

                sample["tracks"] = tracks[:, sampled_indices, :]
                sample["track_vis_mask"] = track_vis_mask[:, sampled_indices]
                sample["track_positive_mask"] = torch.ones(sample["track_vis_mask"].shape[1]).bool()

            else:
                # Generate tracks on-the-fly using the final processed Tensors.
                tracks, track_vis_mask, track_positive_mask = build_tracks_by_depth_pano(
                    sample["extrinsics"],
                    sample["world_points"],
                    sample["depths"],
                    sample["point_masks"],
                    sample["images"],
                    target_track_num=self.track_num,
                    seq_name=sample["seq_name"]
                )
                sample["tracks"] = tracks
                sample["track_vis_mask"] = track_vis_mask
                sample["track_positive_mask"] = track_positive_mask

        # --- Stage 4: Convert lists -> tensors (final unification) ---
        for key, value in sample.items():
            if isinstance(value, list):
                if all(isinstance(v, torch.Tensor) for v in value):
                    sample[key] = torch.stack(value)
                elif all(isinstance(v, np.ndarray) for v in value):
                    sample[key] = torch.from_numpy(np.stack(value))

        return sample


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """
    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]

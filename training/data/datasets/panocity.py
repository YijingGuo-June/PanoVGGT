import os
import os.path as osp
import json
import logging
import random
import math
import time

import cv2
import numpy as np
import torch
from PIL import Image  # Header-only image metadata reading.

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
from panovggt.Projection import EquirecRotate
from training.data.cache_utils import load_or_build_json_cache


class PanoCityDataset(BaseDataset):
    """
    Dataset loader for the PanoCity dataset.
    Each split (typically 24 consecutive frames) from splits_config.json is treated as a trajectory.
    """

    def __init__(
            self,
            common_conf,
            split: str = "train",
            PanoCity_DIR: str = "/data/dataset/panorama/panocity",
            splits_config_file: str = "splits_config.json",
            min_num_images: int = 2,
            len_train: int = 100000,
            len_test: int = 10000,
            expand_ratio: int = 3,
            augmentation: dict = None,
            get_nearby: bool = None,  # If None, use common_conf.get_nearby
            split_seed: int = 42      
    ):
        super().__init__(common_conf=common_conf)

        # Limit OpenCV worker threads to avoid dataloader oversubscription.
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

        self.training = common_conf.training
        self.get_nearby = get_nearby if get_nearby is not None else common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.expand_ratio = expand_ratio
        self.PanoCity_DIR = PanoCity_DIR
        self.splits_config_file = splits_config_file
        self.min_num_images = min_num_images
        self.split = split
        self.split_seed = int(split_seed)

        # Split semantics:
        # - split="train"      -> mode="train" (90%)
        # - split="val"/"test" -> mode="val"   (5%)
        # - split="test_final" -> mode="test"  (5%)
        if split == "train":
            self.mode = "train"
            self.dataset_length = len_train
        elif split in ("val", "test"):
            self.mode = "val"
            self.dataset_length = len_test
        elif split == "test_final":
            self.mode = "test"
            self.dataset_length = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"PanoCity_DIR is {self.PanoCity_DIR}")

        self.augmentation = augmentation if augmentation is not None else common_conf.augs

        
        self._poses_cache = {}   # poses_file_path -> dict[name->c2w]
        self._equi_cache = {}    # equ_h -> EquirecRotate

        t0 = time.time()
        self._load_splits_cache()
        logging.info(f"Loaded splits index in {time.time()-t0:.1f}s")

        if self.trajectories:
            # Convert cached JSON list to tuple for internal use.
            self.base_resolution = tuple(self.trajectories[0]['resolution'])
        else:
            self.base_resolution = (512, 1024)
            logging.warning("No trajectories found, using default resolution (512, 1024).")

        self.depth_max = 240.0

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: PanoCity Data size: {len(self.trajectories)} trajectories")
        logging.info(f"{status}: PanoCity Data dataset length: {len(self)}")

    def __len__(self):
        return self.dataset_length

    # ------------------------- cache / indexing -------------------------
    def _load_splits_cache(self):
        """Load split index from cache, or build once and cache it."""
        cache_dir = osp.join(self.PanoCity_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, f"PanoCity_{self.mode}_index.json")

        def build_fn():
            return self._build_index_json()

        self.trajectories = load_or_build_json_cache(cache_path, build_fn)
        self.sequence_list_len = len(self.trajectories)
        if self.trajectories:
            self.base_resolution = tuple(self.trajectories[0]['resolution'])

    
    def _split_indices_path(self):
        """Return the cache path for deterministic split indices."""
        cache_dir = osp.join(self.PanoCity_DIR, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return osp.join(cache_dir, f"fixed_split_indices_seed{self.split_seed}.json")

    def _load_fixed_split_indices(self):
        """Load fixed split indices from disk; return None if missing/invalid."""
        path = self._split_indices_path()
        if not osp.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            for k in ("train", "val", "test", "n_total", "seed", "ratio"):
                if k not in data:
                    logging.warning(f"[SplitFixed] Missing key '{k}' in {path}")
                    return None
            return data
        except Exception as e:
            logging.warning(f"[SplitFixed] Failed to read {path}: {e}")
            return None

    def _save_fixed_split_indices(self, data: dict):
        """Atomically save deterministic split indices to disk."""
        path = self._split_indices_path()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        logging.info(f"[SplitFixed] Saved fixed split to {path}")

    def _get_or_create_fixed_split(self, n_total: int):
        """
        Load an existing fixed split or create one with deterministic seed.

        Ensures consistent 90/5/5 train/val/test partition across runs.
        """
        data = self._load_fixed_split_indices()
        if data is not None:
            
            if int(data.get("n_total", -1)) != n_total:
                logging.warning(f"[SplitFixed] n_total changed ({data['n_total']} -> {n_total}). "
                                f"Still using existing fixed split as requested.")
            return data

        
        rng = random.Random(self.split_seed)
        indices = list(range(n_total))
        rng.shuffle(indices)

        n_train = int(n_total * 0.9)
        n_val = int(n_total * 0.05)
        n_test = n_total - n_train - n_val  # remainder goes to test split

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        data = {
            "seed": self.split_seed,
            "ratio": [0.9, 0.05, 0.05],
            "n_total": n_total,
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }
        self._save_fixed_split_indices(data)
        return data

    # ------------------------- index build -------------------------
    def _build_index_json(self):
        """Build trajectory index records as list[dict]."""
        splits_path = osp.join(self.PanoCity_DIR, self.splits_config_file)
        if not osp.exists(splits_path):
            raise ValueError(f"Splits config file not found at {splits_path}")
        with open(splits_path, 'r') as f:
            splits_data = json.load(f)

        all_splits = splits_data.get('splits', [])
        total_splits = splits_data.get('total_splits', len(all_splits))
        logging.info(f"Loaded {len(all_splits)} total splits from {splits_path}")
        logging.info(f"Dataset reports {total_splits} total splits")

        
        n_total = len(all_splits)
        fixed = self._get_or_create_fixed_split(n_total)
        train_indices = fixed["train"]
        val_indices = fixed["val"]
        test_indices = fixed["test"]

        
        if self.mode == "train":
            selected_indices = train_indices
        elif self.mode == "val":
            selected_indices = val_indices
        elif self.mode == "test":
            selected_indices = test_indices
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        selected_splits = [all_splits[i] for i in selected_indices]
        logging.info(
            f"Selected {len(selected_splits)} splits for {self.mode} "
            f"({len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test) "
            f"[fixed seed={self.split_seed}]"
        )

        trajs = []
        base_resolution = None

        for split_info in selected_splits:
            scene = split_info['scene']
            block = split_info['block']
            part_id = split_info['part_id']
            pano_images = split_info['pano_images']
            panodepth_images = split_info['panodepth_images']
            poses_file = split_info['poses_file']
            idxs = split_info['indices']

            if len(pano_images) < self.min_num_images:
                continue

            first_rgb = osp.join(self.PanoCity_DIR, pano_images[0])
            if not osp.exists(first_rgb):
                logging.debug(f"First image missing: {first_rgb}")
                continue

            if base_resolution is None:
                try:
                    with Image.open(first_rgb) as img:
                        W, H = img.size
                    base_resolution = (H, W)
                except Exception as e:
                    logging.warning(f"Read header failed {first_rgb}: {e}")
                    continue

            trajs.append({
                'scene': scene,
                'block': block,
                'part_id': part_id,
                'pano_images': pano_images,
                'panodepth_images': panodepth_images,
                'poses_file': poses_file,
                'indices': idxs,
                'resolution': list(base_resolution)  # JSON-serializable
            })

        return trajs
    

    def _read_poses(self, poses_file_path):
        """Read NeRF-style c2w poses from JSON and cache by file path."""
        if poses_file_path in self._poses_cache:
            return self._poses_cache[poses_file_path]
        try:
            with open(poses_file_path, 'r') as f:
                poses_data = json.load(f)
            frames = poses_data.get('frames', [])
            poses_dict = {}
            for frame_data in frames:
                img_name = frame_data.get('name')
                tm = frame_data.get('transformation_matrix')
                if (img_name is None) or (tm is None):
                    continue
                c2w = np.array(tm, dtype=np.float32)
                if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                    continue
                poses_dict[img_name] = c2w
            self._poses_cache[poses_file_path] = poses_dict
            return poses_dict
        except Exception as e:
            logging.error(f"Error reading poses from {poses_file_path}: {e}")
            return {}

    @staticmethod
    def _c2w_to_w2c(pose_c2w):
        if pose_c2w is None:
            return None
        R_c2w = pose_c2w[:3, :3]
        t_c2w = pose_c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        pose_w2c = np.zeros((3, 4), dtype=np.float32)
        pose_w2c[:3, :3] = R_w2c
        pose_w2c[:3, 3] = t_w2c
        return pose_w2c

    def _get_equi_rotate(self, equ_h: int):
        rot = self._equi_cache.get(equ_h, None)
        if rot is None:
            rot = EquirecRotate(equ_h)  # Keep on CPU; moved to target device in use.
            self._equi_cache[equ_h] = rot
        return rot

    def _prepare_augmentation_params(self):
        if not self.training or not self.augmentation:
            return None

        def sample_angle(key, default=60.0):
            if key in self.augmentation and random.random() > 0.5:
                A = float(self.augmentation[key].get('sample_angle', default))
                return (np.random.rand() - 0.5) * 2.0 * A
            return 0.0

        pitch_deg = sample_angle('pitch')
        yaw_deg   = sample_angle('yaw')
        roll_deg  = sample_angle('roll')

        if abs(pitch_deg) < 1e-6 and abs(yaw_deg) < 1e-6 and abs(roll_deg) < 1e-6:
            return None

        ax = math.radians(pitch_deg)
        ay = math.radians(yaw_deg)
        az = math.radians(roll_deg)

        def Rx(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor([[1, 0, 0],
                                 [0, c, -s],
                                 [0, s,  c]], dtype=torch.float32)

        def Ry(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor([[ c, 0, s],
                                 [ 0, 1, 0],
                                 [-s, 0, c]], dtype=torch.float32)

        def Rz(a):
            c, s = math.cos(a), math.sin(a)
            return torch.tensor([[ c, -s, 0],
                                 [ s,  c, 0],
                                 [ 0,  0, 1]], dtype=torch.float32)

        return (Rz(az) @ Ry(ay) @ Rx(ax))

    def _read_and_resize_image(self, path, target_resolution):
        """Read and resize an RGB panorama using OpenCV."""
        h, w = target_resolution
        try:
            img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise IOError(f"cv2.imread failed: {path}")
            if img_bgr.shape[:2] != (h, w):
                img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
        except Exception as e:
            logging.error(f"Error reading image {path}: {e}")
            return np.zeros((3, h, w), dtype=np.float32)

    def _to_single_channel(self, d: np.ndarray) -> np.ndarray:
        """Ensure depth input is a single-channel 2D array of shape (H, W)."""
        if d.ndim == 3:
            # Handle color-encoded PNG depth maps by converting to grayscale.
            if d.shape[2] == 3:
                d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
            elif d.shape[2] == 4:
                d = cv2.cvtColor(d, cv2.COLOR_BGRA2GRAY)
            else:
                d = d[..., 0]  
        return d

    def _read_and_resize_depth(self, path, target_resolution):
        """Read/resize depth map, convert cm to meters, return shape (1, H, W)."""
        h, w = target_resolution
        try:
            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if d is None:
                raise IOError(f"cv2.imread failed: {path}")

            d = self._to_single_channel(d)  

            if d.shape[:2] != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)

            img = d.astype(np.float32) / 100.0  # cm -> m
            img = threshold_depth_map(img, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)
            img[~np.isfinite(img)] = 0.0
            return img[None, ...]  # (1, H, W)
        except Exception as e:
            logging.error(f"Error reading depth {path}: {e}")
            return np.zeros((1, h, w), dtype=np.float32)

    def get_data(
            self,
            seq_index: int = None,
            img_per_seq: int = None,
            seq_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
    ) -> dict:
        """Retrieve data for a specific trajectory."""
        if self.sequence_list_len == 0:
            raise RuntimeError("No trajectories available. Check your splits_config.json and data paths.")

        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        traj = self.trajectories[seq_index % self.sequence_list_len]
        scene = traj['scene']; block = traj['block']; part_id = traj['part_id']
        pano_images = traj['pano_images']; panodepth_images = traj['panodepth_images']
        poses_file = traj['poses_file']; indices = traj['indices']
        orig_resolution = tuple(traj['resolution'])

        # Load all poses for this trajectory once.
        poses_file_path = osp.join(self.PanoCity_DIR, poses_file)
        all_poses_dict = self._read_poses(poses_file_path)
        if not all_poses_dict:
            logging.error(f"No poses loaded from {poses_file_path}. Skipping.")
            return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        # Keep only frames with valid finite poses.
        valid_frame_ids = []
        for i in range(len(pano_images)):
            img_filename = osp.basename(pano_images[i])
            if img_filename in all_poses_dict:
                pose = all_poses_dict[img_filename]
                if pose is not None and np.isfinite(pose).all():
                    valid_frame_ids.append(i)
        if len(valid_frame_ids) < 2:
            logging.error(f"Not enough valid poses in {scene}/{block}/part_{part_id}. "
                          f"Found {len(valid_frame_ids)} valid poses out of {len(pano_images)} images.")
            return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        
        if img_per_seq is None:
            max_frames = min(24, len(valid_frame_ids))
            img_per_seq = random.randint(2, max_frames)
        if ids is None:
            ids = np.random.choice(valid_frame_ids, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(pano_images), expand_ratio=self.expand_ratio)
            ids = [int(i) for i in ids if int(i) in valid_frame_ids]
            if len(ids) < 2:
                ids = np.random.choice(valid_frame_ids, max(2, img_per_seq),
                                       replace=self.allow_duplicate_img).tolist()

        
        unique_ids, seen = [], set()
        for idx in ids:
            if idx not in seen:
                unique_ids.append(idx)
                seen.add(idx)

        # Use fixed training resolution.
        target_h, target_w = 518, 1036
        target_resolution = (target_h, target_w)
        equi_rotate = self._get_equi_rotate(target_resolution[0])

        
        processed_frames = {}

        for idx in unique_ids:
            rgb_rel_path = pano_images[idx]; depth_rel_path = panodepth_images[idx]
            img_filename = osp.basename(rgb_rel_path)
            rgb_path = osp.join(self.PanoCity_DIR, rgb_rel_path)
            depth_path = osp.join(self.PanoCity_DIR, depth_rel_path)

            if not osp.exists(rgb_path) or not osp.exists(depth_path):
                logging.debug(f"Missing RGB/Depth: {rgb_path} / {depth_path}")
                continue
            if img_filename not in all_poses_dict:
                logging.debug(f"Pose not found for {img_filename}")
                continue

            pose_c2w = all_poses_dict[img_filename]

            try:
                image = self._read_and_resize_image(rgb_path, target_resolution)
                depth_map = self._read_and_resize_depth(depth_path, target_resolution)
                pose_w2c = self._c2w_to_w2c(pose_c2w)

                    # Optional per-sample augmentation rotation.
                R_delta = self._prepare_augmentation_params()  # torch 3x3 or None

                frame_data = self.process_one_image(
                    image=image,
                    depth_map=depth_map,
                    extrinsic_w2c=pose_w2c,
                    shape=target_resolution,
                    equi_rotate=equi_rotate,
                    R_delta=R_delta,
                    depth_max=self.depth_max
                )

                processed_frames[idx] = {
                    'rgb': frame_data['rgb'],
                    'depth_tensor': frame_data['depth_tensor'],
                    'extrinsic': frame_data['extrinsic'],
                    'cam_coords': frame_data['cam_coords'],
                    'world_coords': frame_data['world_coords'],
                    'valid_mask': frame_data['valid_mask'],
                    'original_size': np.array(orig_resolution)
                }
            except Exception as e:
                logging.warning(f"Error processing frame {img_filename}: {e}")
                continue

        # batch
        batch_data = {k: [] for k in
                      ['images', 'depths', 'extrinsics', 'cam_points', 'world_points',
                       'point_masks', 'original_sizes']}
        successful_ids = []

        for idx in ids:
            if idx in processed_frames:
                frame = processed_frames[idx]
                batch_data['images'].append(frame['rgb'])
                batch_data['depths'].append(frame['depth_tensor'])
                batch_data['extrinsics'].append(frame['extrinsic'])
                batch_data['cam_points'].append(frame['cam_coords'])
                batch_data['world_points'].append(frame['world_coords'])
                batch_data['point_masks'].append(frame['valid_mask'])
                batch_data['original_sizes'].append(frame['original_size'])
                successful_ids.append(idx)
            else:
                img_filename = osp.basename(pano_images[idx])
                logging.debug(f"Frame {img_filename} (list idx {idx}) was not processed, skipping")

        if len(batch_data['images']) < 2:
            logging.error(f"Not enough valid frames after processing. Retrying...")
            return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        return {
            "seq_name": f"PanoCity_{scene}_{block}_part{part_id}",
            "ids": successful_ids,
            "frame_num": len(batch_data['extrinsics']),
            **batch_data
        }

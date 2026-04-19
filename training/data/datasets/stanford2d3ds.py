import os
import os.path as osp
import glob
import json
import logging
import random
import math
import time

import cv2
import numpy as np
import torch
from PIL import Image  

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import threshold_depth_map
from training.data.cache_utils import load_or_build_json_cache
from panovggt.Projection import EquirecRotate


class Stanford2D3DSDataset(BaseDataset):
    """
    Dataset loader for the Stanford 2D-3D-S dataset.
    Region-based sampling.

    - Camera coords: X-right, Y-down, Z-forward (OpenCV)
    - World coords (file): X-right, Y-forward, Z-up, convert to OpenCV world
    - Poses: 3x4 w2c (world-to-camera)
    - Depth: meters = value / 512.0
    """

    def __init__(
            self,
            common_conf,
            split: str = "train",
            Stanford2D3DS_DIR: str = "/data/dataset/panorama/2d3ds/no_xyz/",
            min_num_images: int = 2,
            len_train: int = 100000,
            len_test: int = 10000,
            expand_ratio: int = 3,
            augmentation: dict = None,
            train_areas: list = None,
            test_areas: list = None,
            test_final_areas: list = None,
            get_nearby: bool = None,
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
        self.Stanford2D3DS_DIR = Stanford2D3DS_DIR
        self.min_num_images = min_num_images
        self.augmentation = augmentation if augmentation is not None else common_conf.augs
        self.split = split

        if train_areas is None:
            train_areas = ['area_1', 'area_2', 'area_3', 'area_4', 'area_6']
        if test_areas is None:
            test_areas = ['area_5a']
        if test_final_areas is None:
            test_final_areas = ['area_5b']

        if split == "train":
            self.dataset_length = len_train
            self.areas = train_areas
            self.mode = "train"
        elif split in ("val", "test"):
            self.dataset_length = len_test
            self.areas = test_areas
            self.mode = "val"
        elif split == "test_final":
            self.dataset_length = len_test
            self.areas = test_final_areas
            self.mode = "test"
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"Stanford2D3DS_DIR is {self.Stanford2D3DS_DIR}")
        logging.info(f"Using areas: {self.areas}")

        
        self._equi_cache = {}

        t0 = time.time()
        self._scan_or_load_trajectories_cache()
        logging.info(f"2D-3D-S index ready in {time.time()-t0:.1f}s")

        if self.trajectories:
            self.base_resolution = self.trajectories[0][4]
        else:
            self.base_resolution = (512, 1024)
            logging.warning("No trajectories found, using default resolution (512, 1024).")

        self.depth_max = 10.0
        self.depth_scale = 512.0  # meters = value / 512.0

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: 2D-3D-S Data size: {len(self.trajectories)} trajectories")
        logging.info(f"{status}: 2D-3D-S Data dataset length: {len(self)}")

    def __len__(self):
        return self.dataset_length

    def _scan_or_load_trajectories_cache(self):
        """
        Load region trajectories from cache, or scan once and write cache.

        Cache path:
            {Stanford2D3DS_DIR}/cache/2d3ds_{self.mode}_index.json
        Record format:
            [area, region_id, room_name, panorama_list(list), [H, W]]
        """
        cache_dir = osp.join(self.Stanford2D3DS_DIR, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, f"2d3ds_{self.mode}_index.json")

        def build_fn():
            trajs = self._scan_trajectories_once()
            serializable = []
            for area, region_id, room_name, panorama_list, res in trajs:
                serializable.append([area, region_id, room_name, list(panorama_list), [res[0], res[1]]])
            return serializable

        data = load_or_build_json_cache(cache_path, build_fn)

        
        self.trajectories = []
        for area, region_id, room_name, panorama_list, res in data:
            self.trajectories.append((area, region_id, room_name, panorama_list, (res[0], res[1])))
        self.sequence_list_len = len(self.trajectories)

    def _scan_trajectories_once(self):
        """
        Scan region trajectories and gather valid panorama IDs.

        Returns:
            (area, region_id, room_name, panorama_list, (H, W))
        Resolution is read from PIL image headers.
        """
        trajectories = []

        for area in self.areas:
            area_path = osp.join(self.Stanford2D3DS_DIR, area)
            if not osp.isdir(area_path):
                logging.warning(f"Area directory not found: {area_path}")
                continue

            room_groups_path = osp.join(area_path, '3d', 'room_groups.json')
            if not osp.exists(room_groups_path):
                logging.warning(f"room_groups.json not found in {area}")
                continue

            try:
                with open(room_groups_path, 'r') as f:
                    room_groups = json.load(f)
            except Exception as e:
                logging.error(f"Error loading room_groups.json from {area}: {e}")
                continue

            for region_key, region_data in room_groups.items():
                region_id = region_data.get('region_id')
                room_name = region_data.get('room_name', f'region_{region_id}')
                panoramas = region_data.get('panoramas', [])

                if len(panoramas) < self.min_num_images:
                    continue

                # Keep panoramas that have RGB, depth, and pose files.
                valid_panoramas = []
                for pano_id in panoramas:
                    rgb_pattern = osp.join(area_path, 'pano', 'rgb',
                                           f'camera_{pano_id}_*_frame_equirectangular_domain_rgb.png')
                    depth_pattern = osp.join(area_path, 'pano', 'depth',
                                             f'camera_{pano_id}_*_frame_equirectangular_domain_depth.png')
                    pose_pattern = osp.join(area_path, 'pano', 'pose',
                                            f'camera_{pano_id}_*_frame_equirectangular_domain_pose.json')
                    if glob.glob(rgb_pattern) and glob.glob(depth_pattern) and glob.glob(pose_pattern):
                        valid_panoramas.append(pano_id)

                if len(valid_panoramas) < self.min_num_images:
                    continue

                try:
                    first_pano = valid_panoramas[0]
                    rgb_pattern = osp.join(area_path, 'pano', 'rgb',
                                           f'camera_{first_pano}_*_frame_equirectangular_domain_rgb.png')
                    rgb_matches = glob.glob(rgb_pattern)
                    if not rgb_matches:
                        logging.warning(f"No RGB file found for {first_pano} in {area}/{room_name}")
                        continue
                    # Read image resolution from header only.
                    with Image.open(rgb_matches[0]) as img:
                        W, H = img.size
                    resolution = (H, W)
                    trajectories.append((area, region_id, room_name, valid_panoramas, resolution))
                except Exception as e:
                    logging.warning(f"Error reading header from {area}/{room_name}: {e}")

        logging.info(f"Found {len(trajectories)} valid region trajectories")
        return trajectories

    def _get_equi_rotate(self, equ_h: int):
        """Get or create a CPU EquirecRotate instance for a target height."""
        rot = self._equi_cache.get(equ_h, None)
        if rot is None:
            rot = EquirecRotate(equ_h)  # Keep on CPU; moved to target device in use.
            self._equi_cache[equ_h] = rot
        return rot

    def get_data(
            self,
            seq_index: int = None,
            img_per_seq: int = None,
            seq_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        area, region_id, room_name, panoramas, orig_resolution = self.trajectories[
            seq_index % self.sequence_list_len]

        # Keep only panorama indices with valid pose files.
        valid_indices = []
        for i, pano_id in enumerate(panoramas):
            pose_pattern = osp.join(self.Stanford2D3DS_DIR, area, 'pano', 'pose',
                                    f'camera_{pano_id}_*_frame_equirectangular_domain_pose.json')
            if glob.glob(pose_pattern):
                valid_indices.append(i)

        if len(valid_indices) < 2:
            logging.error(f"Not enough valid poses in {area}/region_{region_id}. Skipping sequence.")
            return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        
        if img_per_seq is None:
            img_per_seq = random.randint(2, min(24, len(valid_indices)))
        if ids is None:
            ids = np.random.choice(valid_indices, img_per_seq, replace=self.allow_duplicate_img)

        
        if self.get_nearby:
            valid_set = set(valid_indices)
            max_trials = 20
            seed_len = len(ids)
            for _ in range(max_trials):
                expanded = self.get_nearby_ids(ids, len(panoramas), expand_ratio=self.expand_ratio)
                target_len = len(expanded)
                filtered = [int(i) for i in expanded if int(i) in valid_set]
                if len(filtered) == target_len:
                    ids = filtered
                    break
                ids = np.random.choice(valid_indices, seed_len, replace=self.allow_duplicate_img).tolist()
            else:
                deficit = target_len - len(filtered)
                if deficit > 0:
                    pad = np.random.choice(valid_indices, deficit, replace=self.allow_duplicate_img).tolist()
                    filtered.extend(pad)
                ids = filtered
            if len(ids) < 2:
                ids = np.random.choice(valid_indices, max(2, img_per_seq), replace=self.allow_duplicate_img).tolist()

        
        base_h, base_w = self.base_resolution
        new_h = int(base_h / math.sqrt(aspect_ratio))
        target_h = (new_h // 8) * 8
        target_w = target_h * 2
        target_h = 518
        target_w = 1036
        target_resolution = (target_h, target_w)

        # CPU-side panorama rotator, reused by target height.
        equi_rotate = self._get_equi_rotate(target_resolution[0])

        # Per-sample containers for stacked outputs.
        batch_data = {k: [] for k in
                      ['images', 'depths', 'extrinsics', 'cam_points', 'world_points',
                       'point_masks', 'original_sizes']}

        successful_ids = []

        for idx in ids:
            idx = int(idx)
            pano_id = panoramas[idx]

            # Locate panorama RGB/depth/pose triplet.
            rgb_pattern = osp.join(self.Stanford2D3DS_DIR, area, 'pano', 'rgb',
                                   f'camera_{pano_id}_*_frame_equirectangular_domain_rgb.png')
            depth_pattern = osp.join(self.Stanford2D3DS_DIR, area, 'pano', 'depth',
                                     f'camera_{pano_id}_*_frame_equirectangular_domain_depth.png')
            pose_pattern = osp.join(self.Stanford2D3DS_DIR, area, 'pano', 'pose',
                                    f'camera_{pano_id}_*_frame_equirectangular_domain_pose.json')

            rgb_matches = glob.glob(rgb_pattern)
            depth_matches = glob.glob(depth_pattern)
            pose_matches = glob.glob(pose_pattern)
            if not (rgb_matches and depth_matches and pose_matches):
                logging.warning(f"Missing files for panorama {pano_id} in {area}/{room_name}, skipping")
                continue

            rgb_path = rgb_matches[0]
            depth_path = depth_matches[0]
            pose_path = pose_matches[0]

            try:
                image = self._read_and_resize_image(rgb_path, target_resolution)
                depth_map = self._read_and_resize_depth(depth_path, target_resolution)
                pose_w2c = self._read_pose(pose_path)  # (3,4) w2c in OpenCV
                if pose_w2c is None or not np.isfinite(pose_w2c).all():
                    logging.warning(f"Invalid pose for {area}/{room_name}/{pano_id}")
                    continue

                # Optional per-frame augmentation rotation.
                R_delta = self._prepare_augmentation_params()

                frame_data = self.process_one_image(
                    image=image,
                    depth_map=depth_map,
                    extrinsic_w2c=pose_w2c,
                    shape=target_resolution,
                    equi_rotate=equi_rotate,
                    R_delta=R_delta,             # consumed in BaseDataset.process_one_image
                    depth_max=self.depth_max
                )

                batch_data['images'].append(frame_data['rgb'])
                batch_data['depths'].append(frame_data['depth_tensor'])
                batch_data['extrinsics'].append(frame_data['extrinsic'])
                batch_data['cam_points'].append(frame_data['cam_coords'])
                batch_data['world_points'].append(frame_data['world_coords'])
                batch_data['point_masks'].append(frame_data['valid_mask'])
                batch_data['original_sizes'].append(np.array(orig_resolution))

                successful_ids.append(idx)
            except Exception as e:
                logging.warning(f"Error processing panorama {area}/{room_name}/{pano_id}: {e}")
                continue

        # if len(batch_data['images']) < 2:
        # logging.error(f"Not enough valid frames after processing in {area}/region_{region_id}. Retrying...")
        # return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        return {
            "seq_name": f"stanford2d3ds_{area}_region{region_id}_{room_name}",
            "ids": successful_ids,
            "frame_num": len(batch_data['extrinsics']),
            **batch_data
        }

    def _read_pose(self, path):
        """
        Read camera pose from JSON file and convert to OpenCV coordinate system.

        2D-3D-S World:  X-right, Y-forward, Z-up
        OpenCV World:   X-right, Y-down,   Z-forward

        Input JSON pose is w2c (3x4) in 2D-3D-S world.
        Output is w2c (3x4) in OpenCV world.
        """
        try:
            with open(path, 'r') as f:
                pose_data = json.load(f)

            w2c_2d3ds = np.array(pose_data['camera_rt_matrix'], dtype=np.float32)
            if w2c_2d3ds.shape != (3, 4):
                logging.error(f"Invalid pose matrix shape {w2c_2d3ds.shape} in {path}")
                return None

            w2c_2d3ds_4x4 = np.eye(4, dtype=np.float32)
            w2c_2d3ds_4x4[:3, :] = w2c_2d3ds

            # 2D-3D-S -> OpenCV
            T_world_conversion = np.array([
                [1, 0, 0, 0],   # X -> X
                [0, 0, -1, 0],  # Y_opencv = -Z_2d3ds
                [0, 1, 0, 0],   # Z_opencv = Y_2d3ds
                [0, 0, 0, 1]
            ], dtype=np.float32)

            # w2c_opencv = w2c_2d3ds @ T_world_conversion^-1
            w2c_opencv_4x4 = w2c_2d3ds_4x4 @ np.linalg.inv(T_world_conversion)
            w2c_opencv = w2c_opencv_4x4[:3, :]
            return w2c_opencv
        except Exception as e:
            logging.error(f"Error reading pose from {path}: {e}")
            return None

    def _prepare_augmentation_params(self):
        """
        Sample panorama augmentation rotation.

        Returns:
            R_delta: torch.float32 tensor of shape (3, 3) on CPU, or None.
        Axis convention:
            OpenCV camera axes (x-right, y-down, z-forward).
        """
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

        # Rotation composition order: X -> Y -> Z.
        R_delta = Rz(az) @ Ry(ay) @ Rx(ax)
        return R_delta

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
        """Read and resize depth map, convert to meters, return shape (1, H, W)."""
        h, w = target_resolution
        try:
            if not hasattr(self, 'depth_scale') or self.depth_scale in (0, None):
                raise ValueError("self.depth_scale  0， __init__ 。")

            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if d is None:
                raise IOError(f"cv2.imread failed: {path}")

            d = self._to_single_channel(d)

            if d.shape[:2] != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)

            img = d.astype(np.float32) / float(self.depth_scale)
            img = threshold_depth_map(img, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)
            img[~np.isfinite(img)] = 0.0
            return img[None, ...]
        except Exception as e:
            logging.error(f"Error reading depth {path}: {e}")
            return np.zeros((1, h, w), dtype=np.float32)
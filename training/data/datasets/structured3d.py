import os
import os.path as osp
import json
import logging
import random
import time
import math

import cv2
import numpy as np
import torch
from PIL import Image  

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
from training.data.cache_utils import load_or_build_json_cache
from panovggt.Projection import EquirecRotate


class Structured3DDataset(BaseDataset):
    """
    Dataset loader for the Structured3D dataset.
    Each scene is treated as a trajectory, with each room's panorama as a frame.
    Randomly samples lighting and configuration for each room.
    """

    def __init__(
            self,
            common_conf,
            split: str = "train",
            Structured3D_DIR: str = "/data/dataset/panorama/Structured3D_Dataset/Structured3D",
            min_num_rooms: int = 2,
            len_train: int = 100000,
            len_test: int = 10000,
            expand_ratio: int = 3,
            augmentation: dict = None,
            get_nearby: bool = None,  # If None, use common_conf.get_nearby
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
        self.Structured3D_DIR = Structured3D_DIR
        self.min_num_rooms = min_num_rooms
        self.augmentation = augmentation if augmentation is not None else common_conf.augs
        self.split = split

        if split == "train":
            self.dataset_length = len_train
            self.mode = "train"
        elif split in ("val", "test"):
            self.dataset_length = len_test
            self.mode = "val"
        elif split == "test_final":
            self.dataset_length = len_test
            self.mode = "test"            
        else:
            raise ValueError(f"Invalid split: {split}")

        logging.info(f"Structured3D_DIR is {self.Structured3D_DIR}")

        # Configuration and lighting options
        self.configurations = ['empty', 'simple', 'full']
        self.lightings = ['coldlight', 'rawlight', 'warmlight']

        
        self._equi_cache = {}

        
        t0 = time.time()
        self._load_scenes()
        self._scan_or_load_scenes_cache()
        logging.info(f"Structured3D index ready in {time.time()-t0:.1f}s")

        if self.scene_trajectories:
            self.base_resolution = self.scene_trajectories[0][2]
        else:
            self.base_resolution = (512, 1024)
            logging.warning("No scene trajectories found, using default resolution (512, 1024).")

        self.depth_max = 10.0

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Structured3D Data size: {len(self.scene_trajectories)} scene trajectories")
        logging.info(f"{status}: Structured3D Data dataset length: {len(self)}")

    def __len__(self):
        return self.dataset_length

    def _load_scenes(self):
        """Load scene list from Structured3D directory."""
        if not osp.exists(self.Structured3D_DIR):
            raise ValueError(f"Structured3D directory not found at {self.Structured3D_DIR}")

        all_dirs = os.listdir(self.Structured3D_DIR)
        scene_dirs = sorted([d for d in all_dirs if d.startswith('scene_') and
                           osp.isdir(osp.join(self.Structured3D_DIR, d))])

        self.scenes = scene_dirs

        split_file = osp.join(self.Structured3D_DIR, f'{self.mode}.txt')
        if osp.exists(split_file):
            with open(split_file, 'r') as f:
                allowed_scenes = set(line.strip() for line in f)
            self.scenes = [s for s in self.scenes if s in allowed_scenes]
            logging.info(f"Loaded {len(self.scenes)} scenes from {split_file}")
        else:
            random.seed(42)
            random.shuffle(self.scenes)
            split_idx = int(len(self.scenes) * 0.9)
            if self.mode == "train":
                self.scenes = self.scenes[:split_idx]
            else:
                self.scenes = self.scenes[split_idx:]
            logging.info(f"No split file found, using default 80/20 split: {len(self.scenes)} scenes for {self.mode}")

    def _scan_or_load_scenes_cache(self):
        """
        Load scene trajectories from cache, or scan once and write cache.

        Cache path:
            {Structured3D_DIR}/cache/structured3d_{mode}_index.json
        Record format:
            [scene, valid_rooms(list), [H, W]]
        """
        cache_dir = osp.join(self.Structured3D_DIR, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, f"structured3d_{self.mode}_index.json")

        def build_fn():
            trajs = self._scan_scenes_once()
            serializable = []
            for scene, valid_rooms, res in trajs:
                serializable.append([scene, list(valid_rooms), [res[0], res[1]]])
            return serializable

        data = load_or_build_json_cache(cache_path, build_fn)

        
        self.scene_trajectories = []
        for scene, valid_rooms, res in data:
            self.scene_trajectories.append((scene, valid_rooms, (res[0], res[1])))
        self.sequence_list_len = len(self.scene_trajectories)

    def _scan_scenes_once(self):
        """Scan all scenes and collect valid rooms using PIL image headers."""
        scene_trajectories = []  # (scene_id, room_list, resolution)

        for scene in self.scenes:
            scene_path = osp.join(self.Structured3D_DIR, scene)
            rendering_path = osp.join(scene_path, '2D_rendering')

            if not osp.isdir(rendering_path):
                logging.warning(f"2D_rendering directory not found for {scene}, skipping")
                continue

            try:
                room_dirs = sorted([d for d in os.listdir(rendering_path)
                                  if osp.isdir(osp.join(rendering_path, d))])
            except Exception as e:
                logging.warning(f"Error listing rooms in {scene}: {e}")
                continue

            valid_rooms = []
            for room_id in room_dirs:
                room_path = osp.join(rendering_path, room_id, 'panorama')
                if not osp.isdir(room_path):
                    continue

                camera_file = osp.join(room_path, 'camera_xyz.txt')
                if not osp.exists(camera_file):
                    continue

                has_valid_data = False
                for config in self.configurations:
                    config_path = osp.join(room_path, config)
                    if not osp.isdir(config_path):
                        continue
                    for lighting in self.lightings:
                        rgb_file = osp.join(config_path, f'rgb_{lighting}.png')
                        depth_file = osp.join(config_path, 'depth.png')
                        if osp.exists(rgb_file) and osp.exists(depth_file):
                            has_valid_data = True
                            break
                    if has_valid_data:
                        break

                if has_valid_data:
                    valid_rooms.append(room_id)

            if len(valid_rooms) < self.min_num_rooms:
                continue

            # Read RGB resolution from image header only.
            try:
                first_room = valid_rooms[0]
                room_path = osp.join(rendering_path, first_room, 'panorama')
                sample_path = None
                for config in self.configurations:
                    config_path = osp.join(room_path, config)
                    if not osp.isdir(config_path):
                        continue
                    for lighting in self.lightings:
                        rgb_file = osp.join(config_path, f'rgb_{lighting}.png')
                        if osp.exists(rgb_file):
                            sample_path = rgb_file
                            break
                    if sample_path is not None:
                        break

                if sample_path is not None:
                    with Image.open(sample_path) as img:
                        W, H = img.size
                    resolution = (H, W)
                    scene_trajectories.append((scene, valid_rooms, resolution))
                else:
                    logging.warning(f"Could not find sample rgb in {scene}/{first_room}")
            except Exception as e:
                logging.warning(f"Error reading header from {scene}: {e}")

        logging.info(f"Found {len(scene_trajectories)} valid scene trajectories")
        return scene_trajectories

    def _get_equi_rotate(self, equ_h: int):
        """Get or create a CPU EquirecRotate instance for a target height."""
        rot = self._equi_cache.get(equ_h, None)
        if rot is None:
            rot = EquirecRotate(equ_h)  # Keep on CPU; moved to target device in use.
            self._equi_cache[equ_h] = rot
        return rot

    def _generate_camera_pose(self, camera_position):
        """
        Generate camera-to-world transformation matrix from camera position.

        Structured3D coordinate system (verified by testing):
        - World coords:  X-right, Y-forward, Z-up
        - Camera coords: X-right, Y-forward, Z-up (SAME as world!)
        - Camera looks along +Y, up is +Z
        - c2w transformation has NO rotation (R = Identity), only translation

        Target OpenCV coordinate system:
        - Camera coords: X-right, Y-down, Z-forward
        - World coords:  X-right, Y-down, Z-forward
        """
        camera_position = np.array(camera_position, dtype=np.float32)

        c2w_s3d = np.eye(4, dtype=np.float32)
        c2w_s3d[:3, 3] = camera_position  # Only translation, rotation is identity

        T_cam_s3d_to_opencv = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        T_world_s3d_to_opencv = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        pose_c2w_opencv = T_world_s3d_to_opencv @ c2w_s3d @ np.linalg.inv(T_cam_s3d_to_opencv)
        return pose_c2w_opencv

    def _read_camera_position(self, path):
        """
        Read camera position from camera_xyz.txt.
        """
        try:
            camera_pos_mm = np.loadtxt(path, dtype=np.float32)
            if camera_pos_mm.shape != (3,):
                logging.error(f"Invalid camera position shape {camera_pos_mm.shape} in {path}")
                return None
            camera_pos_m = camera_pos_mm / 1000.0
            return camera_pos_m
        except Exception as e:
            logging.error(f"Error reading camera position from {path}: {e}")
            return None

    @staticmethod
    def _c2w_to_w2c(pose_c2w):
        """Convert camera-to-world matrix to world-to-camera matrix."""
        R_c2w = pose_c2w[:3, :3]
        t_c2w = pose_c2w[:3, 3]
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ t_c2w
        pose_w2c = np.zeros((3, 4), dtype=np.float32)
        pose_w2c[:3, :3] = R_w2c
        pose_w2c[:3, 3] = t_w2c
        return pose_w2c

    def _select_room_variant(self, room_path):
        """
        Randomly select a valid configuration and lighting for a room.
        Returns: (config, lighting, rgb_path, depth_path) or (None, None, None, None)
        """
        configs = self.configurations.copy()
        lightings = self.lightings.copy()
        if self.training:
            random.shuffle(configs)
            random.shuffle(lightings)

        for config in configs:
            config_path = osp.join(room_path, config)
            if not osp.isdir(config_path):
                continue
            for lighting in lightings:
                rgb_file = osp.join(config_path, f'rgb_{lighting}.png')
                depth_file = osp.join(config_path, 'depth.png')
                if osp.exists(rgb_file) and osp.exists(depth_file):
                    return config, lighting, rgb_file, depth_file

        return None, None, None, None

    def get_data(
            self,
            seq_index: int = None,
            img_per_seq: int = None,
            seq_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
    ) -> dict:
        """Retrieve data for a specific scene sequence."""
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        scene, valid_rooms, orig_resolution = self.scene_trajectories[
            seq_index % self.sequence_list_len]

        scene_path = osp.join(self.Structured3D_DIR, scene)
        rendering_path = osp.join(scene_path, '2D_rendering')

        
        room_data = []  # (room_id, pose_c2w, config, lighting, rgb_path, depth_path)

        for room_id in valid_rooms:
            room_path = osp.join(rendering_path, room_id, 'panorama')

            camera_file = osp.join(room_path, 'camera_xyz.txt')
            try:
                camera_pos = self._read_camera_position(camera_file)
                if camera_pos is None or not np.isfinite(camera_pos).all():
                    continue

                pose_c2w = self._generate_camera_pose(camera_pos)
                if not np.isfinite(pose_c2w).all():
                    continue

                config, lighting, rgb_path, depth_path = self._select_room_variant(room_path)
                if config is None:
                    continue

                room_data.append((room_id, pose_c2w, config, lighting, rgb_path, depth_path))
            except Exception as e:
                logging.warning(f"Error processing room {scene}/{room_id}: {e}")
                continue

        if len(room_data) < 2:
            logging.error(f"Not enough valid rooms in {scene}. Skipping sequence.")
            return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        
        if img_per_seq is None:
            max_frames = min(24, len(room_data))
            img_per_seq = random.randint(2, max_frames)
        if ids is None:
            ids = np.random.choice(len(room_data), img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(room_data), expand_ratio=self.expand_ratio)
            ids = [int(i) for i in ids if int(i) < len(room_data)]
            if len(ids) < 2:
                ids = np.random.choice(len(room_data), max(2, img_per_seq),
                                       replace=self.allow_duplicate_img).tolist()

        # Use fixed training resolution.
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
        selected_room_info = []
        successful_ids = []

        for idx in ids:
            idx = int(idx)
            room_id, pose_c2w, config, lighting, rgb_path, depth_path = room_data[idx]

            try:
                image = self._read_and_resize_image(rgb_path, target_resolution)
                depth_map = self._read_and_resize_depth(depth_path, target_resolution)

                # Convert c2w to w2c (3x4).
                pose_w2c = self._c2w_to_w2c(pose_c2w)

                # Optional per-frame augmentation rotation.
                R_delta = self._prepare_augmentation_params()

                frame_data = self.process_one_image(
                    image=image,
                    depth_map=depth_map,
                    extrinsic_w2c=pose_w2c,
                    shape=target_resolution,
                    equi_rotate=equi_rotate,
                    R_delta=R_delta,          # per-sample augmentation
                    depth_max=self.depth_max
                )

                batch_data['images'].append(frame_data['rgb'])
                batch_data['depths'].append(frame_data['depth_tensor'])
                batch_data['extrinsics'].append(frame_data['extrinsic'])
                batch_data['cam_points'].append(frame_data['cam_coords'])
                batch_data['world_points'].append(frame_data['world_coords'])
                batch_data['point_masks'].append(frame_data['valid_mask'])
                batch_data['original_sizes'].append(np.array(orig_resolution))

                selected_room_info.append(f"{room_id}_{config}_{lighting}")
                successful_ids.append(idx)
            except Exception as e:
                logging.warning(f"Error processing room {room_id}: {e}")
                continue

        if len(batch_data['images']) < 2:
            logging.error(f"Not enough valid frames after processing. Retrying...")
            return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        return {
            "seq_name": f"structured3d_{scene}_{'_'.join(selected_room_info[:3])}",
            "ids": successful_ids,
            "frame_num": len(batch_data['extrinsics']),
            **batch_data
        }

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
            return (img_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)  # CHW
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
        """Read and resize depth map, convert mm to meters, return shape (1, H, W)."""
        h, w = target_resolution
        try:
            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if d is None:
                raise IOError(f"cv2.imread failed: {path}")

            d = self._to_single_channel(d)

            if d.shape[:2] != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)

            img = d.astype(np.float32) / 1000.0  # mm -> m
            img = threshold_depth_map(img, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)
            img[~np.isfinite(img)] = 0.0
            return img[None, ...]
        except Exception as e:
            logging.error(f"Error reading depth {path}: {e}")
            return np.zeros((1, h, w), dtype=np.float32)
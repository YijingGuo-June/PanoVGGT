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
from PIL import Image

from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
from panovggt.Projection import EquirecRotate
from training.data.cache_utils import load_or_build_json_cache


class Matterport3DDataset(BaseDataset):
    """
    Dataset loader for the Matterport3D dataset.
    Sampling is done at room level instead of scene level.
    """

    def __init__(
            self,
            common_conf,
            split: str = "train",
            Matterport3D_DIR: str = "/data/dataset/panorama/matterport3D",
            min_num_images: int = 2,
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
        self.Matterport3D_DIR = Matterport3D_DIR
        self.min_num_images = min_num_images
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

        logging.info(f"Matterport3D_DIR is {self.Matterport3D_DIR}")

        # Cache CPU-side EquirecRotate instances by panorama height.
        self._equi_cache = {}

        t0 = time.time()
        self._load_scenes()
        self._scan_or_load_rooms_cache()
        logging.info(f"Matterport3D index ready in {time.time()-t0:.1f}s")

        if self.room_trajectories:
            self.base_resolution = self.room_trajectories[0][4]
        else:
            self.base_resolution = (512, 1024)  # Default resolution
            logging.warning("No room trajectories found, using default resolution (512, 1024).")

        self.depth_max = 10.0

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: Matterport3D Data size: {len(self.room_trajectories)} room trajectories")
        logging.info(f"{status}: Matterport3D Data dataset length: {len(self)}")

    def __len__(self):
        return self.dataset_length

    def _load_scenes(self):
        """Load scene list from parsed_json directory."""
        parsed_json_dir = osp.join(self.Matterport3D_DIR, 'parsed_json')
        if not osp.exists(parsed_json_dir):
            raise ValueError(f"parsed_json directory not found at {parsed_json_dir}")

        json_files = [f for f in os.listdir(parsed_json_dir) if f.endswith('.json')]
        self.scenes = [f[:-5] for f in json_files]  # remove .json

        split_file = osp.join(self.Matterport3D_DIR, f'{self.mode}.txt')
        if osp.exists(split_file):
            with open(split_file, 'r') as f:
                allowed_scenes = set(line.strip() for line in f)
            self.scenes = [s for s in self.scenes if s in allowed_scenes]

        logging.info(f"Loaded {len(self.scenes)} scenes for {self.mode} split")

    def _scan_or_load_rooms_cache(self):
        """
        Load room trajectories from cache, or scan once and write cache.

        Cache path:
            {Matterport3D_DIR}/cache/matterport3d_{mode}_index.json
        Record format:
            (scene, room_id, room_name, valid_views, (H, W))
        """
        cache_dir = osp.join(self.Matterport3D_DIR, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, f"matterport3d_{self.mode}_index.json")

        def build_fn():
            trajs = self._scan_rooms_once()  # list of tuples
            # Serialize tuples to JSON-friendly lists.
            serializable = []
            for scene, room_id, room_name, valid_views, res in trajs:
                serializable.append([scene, room_id, room_name, valid_views, [res[0], res[1]]])
            return serializable

        data = load_or_build_json_cache(cache_path, build_fn)
        
        self.room_trajectories = []
        for scene, room_id, room_name, valid_views, res in data:
            self.room_trajectories.append((scene, room_id, room_name, valid_views, (res[0], res[1])))
        self.sequence_list_len = len(self.room_trajectories)

    def _scan_rooms_once(self):
        """Scan all rooms in all scenes using parsed JSON and PIL image headers."""
        room_trajectories = []  # (scene, room_id, room_name, valid_views, (H,W))
        parsed_json_dir = osp.join(self.Matterport3D_DIR, 'parsed_json')

        for scene in self.scenes:
            scene_path = osp.join(self.Matterport3D_DIR, scene)
            json_path = osp.join(parsed_json_dir, f'{scene}.json')
            if not osp.exists(json_path):
                logging.warning(f"JSON file not found for scene {scene}, skipping")
                continue

            with open(json_path, 'r') as f:
                room_info = json.load(f)

            color_dir = osp.join(scene_path, 'pano_skybox_color')
            depth_dir = osp.join(scene_path, 'pano_depth')
            pose_dir = osp.join(scene_path, 'pano_poses')

            if not all(osp.isdir(d) for d in [color_dir, depth_dir, pose_dir]):
                alt_color = osp.join(scene_path, 'pano_color')
                if not all(osp.isdir(d) for d in [alt_color, depth_dir, pose_dir]):
                    logging.warning(f"Missing required directories for scene {scene}, skipping")
                    continue
                color_dir = alt_color

            for room_id, room_data in room_info.items():
                room_name = room_data.get('room_name', f'room_{room_id}')
                panoramas = room_data.get('panoramas', [])
                if len(panoramas) < self.min_num_images:
                    continue

                valid_views = []
                for pano_id in panoramas:
                    cpath_png = osp.join(color_dir, f"{pano_id}.png")
                    cpath_jpg = osp.join(color_dir, f"{pano_id}.jpg")
                    cpath = cpath_png if osp.exists(cpath_png) else cpath_jpg
                    dpath = osp.join(depth_dir, f"{pano_id}.png")
                    ppath = osp.join(pose_dir, f"{pano_id}.txt")
                    if cpath and osp.exists(cpath) and osp.exists(dpath) and osp.exists(ppath):
                        valid_views.append(pano_id)

                if len(valid_views) < self.min_num_images:
                    continue

                # Read image size from header only (fast metadata access).
                try:
                    first_view = valid_views[0]
                    cpath_png = osp.join(color_dir, f"{first_view}.png")
                    cpath_jpg = osp.join(color_dir, f"{first_view}.jpg")
                    cpath = cpath_png if osp.exists(cpath_png) else cpath_jpg
                    with Image.open(cpath) as img:
                        W, H = img.size
                    resolution = (H, W)
                    room_trajectories.append((scene, room_id, room_name, valid_views, resolution))
                except Exception as e:
                    logging.warning(f"Error reading header from {scene}/{room_id}: {e}")

        logging.info(f"Found {len(room_trajectories)} valid room trajectories")
        return room_trajectories

    def _get_equi_rotate(self, equ_h: int):
        """Get or create a CPU EquirecRotate instance for a target height."""
        rot = self._equi_cache.get(equ_h, None)
        if rot is None:
            rot = EquirecRotate(equ_h)  # Keep on CPU; moved to target device in use.
            self._equi_cache[equ_h] = rot
        return rot

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
        """Read and resize depth map, convert to meters, and return shape (1, H, W)."""
        h, w = target_resolution
        try:
            d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if d is None:
                raise IOError(f"cv2.imread failed: {path}")

            d = self._to_single_channel(d)

            if d.shape[:2] != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)

            img = d.astype(np.float32) / 4000.0
            img = threshold_depth_map(img, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)
            img[~np.isfinite(img)] = 0.0
            return img[None, ...]
        except Exception as e:
            logging.error(f"Error reading depth {path}: {e}")
            return np.zeros((1, h, w), dtype=np.float32)
        

    def _read_pose(self, path):
        """
        Read camera pose file and convert to OpenCV coordinate system.

        Coordinate systems:
        - Matterport3D world:  X-right, Y-forward, Z-up
        - Matterport3D camera: X-right, Y-up, Z-backward (OpenGL convention)
        - OpenCV world:        X-right, Y-down, Z-forward
        - OpenCV camera:       X-right, Y-down, Z-forward
        """
        try:
            pose_c2w_mp3d = np.loadtxt(path, dtype=np.float32)
            if pose_c2w_mp3d.shape != (4, 4):
                logging.error(f"Invalid pose matrix shape {pose_c2w_mp3d.shape} in {path}")
                return None

            T_cam_mp3d_to_opencv = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            T_world_mp3d_to_opencv = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            pose_c2w_opencv = T_world_mp3d_to_opencv @ pose_c2w_mp3d @ np.linalg.inv(T_cam_mp3d_to_opencv)
            return pose_c2w_opencv
        except Exception as e:
            logging.error(f"Error reading pose from {path}: {e}")
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
        pose_w2c[:3, 3]  = t_w2c
        return pose_w2c

    def get_data(
            self,
            seq_index: int = None,
            img_per_seq: int = None,
            seq_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
    ) -> dict:
        """Retrieve data for a specific room sequence."""
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_index is None:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        scene, room_id, room_name, valid_views, orig_resolution = self.room_trajectories[
            seq_index % self.sequence_list_len]

        # Frame count is based on valid views with RGB, depth, and pose.
        frame_count = len(valid_views)
        if img_per_seq is None:
            max_frames = min(24, frame_count)
            img_per_seq = random.randint(2, max_frames)
        if ids is None:
            ids = np.random.choice(frame_count, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, frame_count, expand_ratio=self.expand_ratio)
            ids = [int(i) for i in ids if 0 <= int(i) < frame_count]
            if len(ids) < 2:
                ids = np.random.choice(frame_count, max(2, img_per_seq), replace=self.allow_duplicate_img).tolist()

        # Use fixed training resolution.
        base_h, base_w = self.base_resolution
        new_h = int(base_h / math.sqrt(aspect_ratio))
        target_h = (new_h // 8) * 8
        target_w = target_h * 2
        target_h = 518
        target_w = 1036
        target_resolution = (target_h, target_w)

        equi_rotate = self._get_equi_rotate(target_resolution[0])

        batch_data = {k: [] for k in
                      ['images', 'depths', 'extrinsics', 'cam_points', 'world_points',
                       'point_masks', 'original_sizes']}

        successful_ids = []
        for idx in ids:
            idx = int(idx)
            view_name = valid_views[idx]

            color_dir = osp.join(self.Matterport3D_DIR, scene, 'pano_skybox_color')
            if not osp.exists(color_dir):
                color_dir = osp.join(self.Matterport3D_DIR, scene, 'pano_color')
            color_png = osp.join(color_dir, f"{view_name}.png")
            color_jpg = osp.join(color_dir, f"{view_name}.jpg")
            color_path = color_png if osp.exists(color_png) else color_jpg

            depth_path = osp.join(self.Matterport3D_DIR, scene, 'pano_depth', f"{view_name}.png")
            pose_path  = osp.join(self.Matterport3D_DIR, scene, 'pano_poses', f"{view_name}.txt")

            try:
                pose_c2w = self._read_pose(pose_path)
                if pose_c2w is None or not np.isfinite(pose_c2w).all():
                    continue
                pose_w2c = self._c2w_to_w2c(pose_c2w)

                image = self._read_and_resize_image(color_path, target_resolution)
                depth_map = self._read_and_resize_depth(depth_path, target_resolution)

                R_delta = self._prepare_augmentation_params()

                frame_data = self.process_one_image(
                    image=image,
                    depth_map=depth_map,
                    extrinsic_w2c=pose_w2c,
                    shape=target_resolution,
                    equi_rotate=equi_rotate,
                    R_delta=R_delta,  # per-sample
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

                if len(successful_ids) >= img_per_seq and not self.allow_duplicate_img:
                    break
            except Exception as e:
                logging.warning(f"Error processing view {scene}/{view_name}: {e}")
                continue

        # if len(batch_data['images']) < 2:
        # logging.error(f"Not enough valid frames after processing in {scene}/room_{room_id}. Retrying...")
        # return self.get_data(img_per_seq=img_per_seq, aspect_ratio=aspect_ratio)

        return {
            "seq_name": f"matterport3d_{scene}_room{room_id}_{room_name}",
            "ids": successful_ids,
            "frame_num": len(batch_data['extrinsics']),
            **batch_data
        }
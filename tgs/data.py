import json
import math
from dataclasses import dataclass, field
import open3d as o3d
import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import skimage.io as io
import skimage.transform
import itertools
from PIL import Image
from torch.utils.data import Dataset

from tgs.utils.config import parse_structured
from tgs.utils.ops import get_intrinsic_from_fov, get_ray_directions, get_rays, get_world_directions
from tgs.utils.typing import *

# Todo: Make this dataloader batch compliant
# Todo: Check for normalization
def _parse_scene_list_single(scene_list_path: str):
    if scene_list_path.endswith(".json"):
        with open(scene_list_path) as f:
            all_scenes = json.loads(f.read())
    elif scene_list_path.endswith(".txt"):
        with open(scene_list_path) as f:
            all_scenes = [p.strip() for p in f.readlines()]
    else:
        all_scenes = [scene_list_path]

    return all_scenes


def _parse_scene_list(scene_list_path: Union[str, List[str]]):
    all_scenes = []
    if isinstance(scene_list_path, str):
        scene_list_path = [scene_list_path]
    for scene_list_path_ in scene_list_path:
        all_scenes += _parse_scene_list_single(scene_list_path_)
    return all_scenes

def _parse_scene_HUMBI(dataset_dir: dict):
    view_list = []
    for subject_no in dataset_dir.keys():
        view_dict = dataset_dir[subject_no]
        for view_no in view_dict.keys():
            if not view_no.isnumeric():
                continue
            view_level_dict = {}
            view_level_dict[subject_no] = subject_no
            view_level_dict['intrinsics'] = dataset_dir[subject_no]['intrinsic']
            view_level_dict['extrinsics'] = dataset_dir[subject_no]['w2c_rot']
            view_level_dict['camera_centers'] = dataset_dir[subject_no]['camera_center']
            view_level_dict['rgb_imgs'] = view_dict[view_no]['imgs']
            view_level_dict['seg_imgs'] = [rgb_name.split('.')[0]+'_rgba' + rgb_name.split('.')[1] for rgb_name in view_level_dict['rgb_imgs']]
            view_level_dict['pointcloud_sparse'] = view_dict[view_no]['pointcloud_sparse']
            view_level_dict['pointcloud_dense'] = view_dict[view_no]['pointcloud_dense']
            for img in view_level_dict['rgb_imgs']:
                cam_idx = int(img.split(".")[0].split("image0")[1])
                view_list.append((img, subject_no, str(cam_idx), view_level_dict))
    return view_list, dataset_dir

@dataclass
class CustomImageDataModuleConfig:
    image_list: Any = ""
    background_color: Tuple[float, float, float] = field(
        default_factory=lambda: (1.0, 1.0, 1.0)
    )

    relative_pose: bool = False
    cond_height: int = 512
    cond_width: int = 512
    cond_camera_distance: float = 1.6
    cond_fovy_deg: float = 40.0
    cond_elevation_deg: float = 0.0
    cond_azimuth_deg: float = 0.0
    num_workers: int = 16

    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    eval_elevation_deg: float = 0.0
    eval_camera_distance: float = 1.6
    eval_fovy_deg: float = 40.0
    n_test_views: int = 120
    num_views_output: int = 120
    only_3dgs: bool = False

class CustomImageOrbitDataset(Dataset):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: CustomImageDataModuleConfig = parse_structured(CustomImageDataModuleConfig, cfg)
        json_file = open('/home/aditya/ML/HUMBI_Dataset/face/extract/dataset.json')
        self.all_scenes, self.dataset_dict = _parse_scene_HUMBI(json.load(json_file))
        json_file.close()

        # About 107 cameras, not all subjects contain images from all cameras. The folder numbers likely represent a particular facial expression for that subject

    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, index):
        img_path, subject_no, cam_idx, view_level_dict = self.all_scenes[index]
        
        def img_load_func(f):
            img = io.imread(f)
            img = skimage.transform.resize(img, (self.cfg.cond_width, self.cfg.cond_height))
            return img
        
        img_cond = torch.from_numpy(
            np.asarray(
                io.ImageCollection(img_path, load_func=img_load_func).concatenate()
            )
        ).float()

        mask_cond: Float[Tensor, "Hc Wc 1"] = img_cond[:, :, :, -1:]
        rgb_cond = img_cond
        subject_int, subject_cc, subject_xform = torch.Tensor(self.dataset_dict[subject_no]['intrinsic'][cam_idx]), torch.Tensor(self.dataset_dict[subject_no]['camera_center'][cam_idx]), torch.Tensor(self.dataset_dict[subject_no]['w2c_rot'][cam_idx])
        world_direction_rays, _ = get_world_directions(subject_int, subject_xform, self.cfg.eval_height, self.cfg.eval_width, subject_cc)
        w2c = torch.cat((torch.cat((subject_xform, (-subject_xform@subject_cc).unsqueeze(1)), dim = 1), torch.zeros((4)).unsqueeze(0)), dim=0)
        w2c[3,3] = 1.0
        c2w = torch.linalg.pinv(w2c)

        #c2w = torch.cat((projection_inverse_rot, torch.zeros_like(projection_inverse_rot[2,:]).unsqueeze(0)), dim=0)
        #c2w = torch.cat((c2w,torch.cat((subject_cc, torch.Tensor([1.0]))).unsqueeze(1)), dim=1)
        #c2w[3,3] = 1.0
        
        self.c2w_cond = c2w
        self.c2w = c2w
        self.intrinsic= subject_int
        self.intrinsic_cond = self.intrinsic
        self.intrinsic_normed_cond = self.intrinsic_cond
        self.intrinsic_normed = self.intrinsic
        self.rays_o = world_direction_rays
        self.rays_d = self.rays_o
        self.camera_positions = subject_cc
        self.background_color = torch.as_tensor(self.cfg.background_color)

        out = {
            "rgb_cond": rgb_cond,
            "c2w_cond": self.c2w_cond.unsqueeze(0),
            "mask_cond": mask_cond,
            "intrinsic_cond": self.intrinsic_cond.unsqueeze(0),
            "intrinsic_normed_cond": self.intrinsic_normed_cond.unsqueeze(0),
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "intrinsic": self.intrinsic.unsqueeze(0),
            "intrinsic_normed": self.intrinsic_normed.unsqueeze(0),
            "c2w": self.c2w.unsqueeze(0),
            "view_index": torch.as_tensor(index).unsqueeze(0),
            "camera_positions": self.camera_positions.unsqueeze(0),
            "pcl_sparse": view_level_dict['pointcloud_sparse'],
            "pcl_dense": view_level_dict['pointcloud_dense'],
            "path": img_path,
        }
        #out["c2w"][..., :3, 1:3] *= -1
        #out["c2w_cond"][..., :3, 1:3] *= -1
        instance_id = (subject_no, cam_idx)
        out["index"] = torch.as_tensor(index)
        out["background_color"] = self.background_color
        out["instance_id"] = instance_id
        return out

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch
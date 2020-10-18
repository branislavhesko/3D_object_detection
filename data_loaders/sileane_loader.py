from enum import auto, Enum
import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform.rotation import Rotation

from config import Config, FeatureConfig
from utils.data_utils import (
    load_model, make_transformation_matrix, find_model_opposite_points, find_negative_matches, find_positive_matches)
from utils.transforms import homogenize_points


class DataEntries(Enum):
    IMAGES = auto()
    DEPTHS = auto()
    GTS = auto()


class SileaneConfig:
    clip_start = None
    clip_end = None
    width = None
    height = None
    fu = None
    fv = None
    cu = None
    cv = None
    rotation = None
    location = None


def depth_to_pointcloud(depth, rgb_image, config: SileaneConfig, transformation_matrix):
    z = config.clip_start[0] + (config.clip_end[0] - config.clip_start[0]) * depth
    h, w, _ = rgb_image.shape
    mesh_x, mesh_y = np.meshgrid(np.arange(w), np.arange(h))
    x = z / config.fu[0] * (mesh_x - config.cu[0])
    y = z / config.fv[0] * (mesh_y - config.cv[0])
    pcd = torch.from_numpy(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1)),
                                           rgb_image.reshape(-1, 3)), axis=1))
    pcd[:, :3] = transform3d(homogenize_points(pcd[:, :3]).float(), transformation_matrix)
    return pcd


class SileaneDataset(Dataset):
    def __init__(self, config: FeatureConfig):
        self._config = config
        self._path = config.path
        self._camera_config = self._load_camera_config()
        self._data = self._load_images_depths()
        self._model = homogenize_points(load_model(config.model_path))
        print(self._data)

    def _load_camera_config(self):
        camera_cfg_path = os.path.join(self._path, "camera_params.txt")
        with open(camera_cfg_path, "r") as cfg:
            lines = cfg.readlines()
        config = SileaneConfig()
        for line in lines:
            if line.split("\t")[0] in dir(config):
                setattr(config, line.split("\t")[0], list(map(float, line.split("\t")[1:])))
        return config

    @staticmethod
    def _make_camera_transform(rot, trans):
        rot = np.array(rot)[[1, 2, 3, 0]]
        rotation_matrix = Rotation.from_quat(rot).as_matrix()
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, -1] = trans
        return torch.from_numpy(transformation_matrix).float()

    def _load_images_depths(self):
        images = sorted(glob.glob(os.path.join(self._path, "rgb", "*.PNG")))
        depths = sorted(glob.glob(os.path.join(self._path, "depth", "*.PNG")))
        gt_pos = sorted(glob.glob(os.path.join(self._path, "gt", "*.json")))
        return {
            DataEntries.IMAGES: images,
            DataEntries.DEPTHS: depths,
            DataEntries.GTS: gt_pos
        }

    def __len__(self):
        return len(self._data[DataEntries.IMAGES])

    def __getitem__(self, item):
        imagef, depthf, gtf = self._data[DataEntries.IMAGES][item], \
                           self._data[DataEntries.DEPTHS][item], \
                           self._data[DataEntries.GTS][item]
        assert os.path.basename(imagef[:-4]) == os.path.basename(depthf)[:-4] == os.path.basename(gtf)[:-5]
        image = cv2.cvtColor(cv2.imread(imagef, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depthf, cv2.IMREAD_UNCHANGED) / (2 ** 16)
        print(f"min: {np.amin(depth)}, max: {np.amax(depth)}")
        pcd = depth_to_pointcloud(depth, image, self._camera_config, self._make_camera_transform(
            self._camera_config.rotation, self._camera_config.location))
        with open(gtf, "r") as file:
            gt = json.load(file)
        gt = [np.expand_dims(make_transformation_matrix(entry), axis=0) for entry in gt]
        gt = np.concatenate(gt, axis=0) if len(gt) else np.expand_dims(np.eye(4), 0)
        gt = torch.from_numpy(gt).float()
        gt_pcd = self._make_gt_pcd(gt)
        if self._config.use_uniform_features:
            pcd_feats = torch.ones(pcd.shape[0])
        else:
            pcd_feats = pcd[:, 3:]
        gt_feats = torch.ones(gt_pcd.shape[0])
        gt_pcd *= self._config.meters_to_millimeters
        pcd *= self._config.meters_to_millimeters
        self._get_pairs(pcd[:, :3], gt_pcd)
        return pcd[:, :3].float(), gt_pcd, pcd_feats, gt_feats, gt

    def _make_gt_pcd(self, gts):
        points = []
        for gt in gts:
            points.append(transform3d(self._model, gt.float()))
        return torch.cat(points, dim=0) if len(points) else torch.zeros(0, 3)

    def _get_pairs(self, pcd, gt):
        pcd_positive_choice_ids = np.random.choice(pcd.shape[0], self._config.num_positive_pairs * 10)
        gt_positive_choice_ids = np.random.choice(gt.shape[0], self._config.num_positive_pairs)
        pcd_negative_choice_ids = np.random.choice(pcd.shape[0], self._config.num_negative_pairs)
        gt_negative_choice_ids = np.random.choice(gt.shape[0], self._config.num_negative_pairs)
        neg_gt, neg_pcd = find_negative_matches(gt[gt_negative_choice_ids, :],
                                                pcd[pcd_negative_choice_ids, :], self._config.limit_positive_distance)
        pos_gt, pos_pcd = find_positive_matches(gt[gt_positive_choice_ids, :], pcd[pcd_positive_choice_ids, :],
                                                self._config.limit_positive_distance)

if __name__ == "__main__":
    import open3d
    from utils.transforms import transform3d
    dataset = SileaneDataset(FeatureConfig())
    model = homogenize_points(load_model("./data/sileane/gear/mesh.ply"))
    for idx in range(10):
        pcd, model_gt, feats_, feats_gt, gt_ = dataset[idx]
        print(model_gt.shape)
        p = open3d.geometry.PointCloud()
        p.points = open3d.utility.Vector3dVector(pcd.numpy()[:, :3])
        pcds = [p]
        for g in gt_:
            k = open3d.geometry.PointCloud()
            k.points = open3d.utility.Vector3dVector(model_gt)
            k.paint_uniform_color([0, 1, 0])
            pcds.append(k)
        open3d.visualization.draw_geometries(pcds)

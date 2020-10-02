from enum import auto, Enum
import glob
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.data_utils import make_transformation_matrix


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


def depth_to_pointcloud(depth, rgb_image, config: SileaneConfig):
    z = config.clip_start + (config.clip_end - config.clip_start) * depth
    h, w, _ = rgb_image.shape
    mesh_x, mesh_y = np.meshgrid(np.arange(w), np.arange(h))
    x = z / config.fu * (mesh_x - config.cu)
    y = z / config.fv * (mesh_y - config.cv)
    return torch.from_numpy(np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1)), z.reshape((-1, 1)),
                                            rgb_image.reshape(-1, 3)), axis=1))


class SileaneDataset(Dataset):
    def __init__(self, path_to_dataset):
        self._path = path_to_dataset
        self._config = self._load_camera_config()
        self._data = self._load_images_depths()
        print(self._data)

    def _load_camera_config(self):
        camera_cfg_path = os.path.join(self._path, "camera_params.txt")
        with open(camera_cfg_path, "r") as cfg:
            lines = cfg.readlines()
        config = SileaneConfig()
        for line in lines:
            if line.split("\t")[0] in dir(config):
                setattr(config, line.split("\t")[0], float(line.split("\t")[1]))
        return config

    def _load_images_depths(self):
        images = sorted(glob.glob(os.path.join(self._path, "rgb", "*.PNG")))
        depths = sorted(glob.glob(os.path.join(self._path, "depth_gt", "*.PNG")))
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
        depth = cv2.imread(depthf, cv2.IMREAD_GRAYSCALE) / 255.
        pcd = depth_to_pointcloud(depth, image, self._config)
        with open(gtf, "r") as file:
            gt = json.load(file)
        gt = [np.expand_dims(make_transformation_matrix(entry), axis=0) for entry in gt]
        gt = np.concatenate(gt, axis=0) if len(gt) else np.empty((0))
        return pcd.float(), torch.from_numpy(gt).float()


if __name__ == "__main__":
    import open3d
    from kaolin.mathutils import transform3d
    dataset = SileaneDataset("./data/sileane/gear")
    for idx in range(10):
        pcd, gt = dataset[idx]
        p = open3d.geometry.PointCloud()
        p.points = open3d.utility.Vector3dVector(pcd.numpy()[:, :3])
        open3d.visualization.draw_geometries([p])
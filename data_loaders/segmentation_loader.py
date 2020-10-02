import glob
import os

import cv2
import torch
from torch.utils.data import Dataset

from config import SegmentationConfig


class SegmentationDataset(Dataset):

    def __init__(self, config: SegmentationConfig, path_to_images, path_to_gt):
        self._config = config
        self._images, self._gts = self._load_images_gt(path_to_images, path_to_gt)
        self._size = cv2.imread(self._images[0], cv2.IMREAD_GRAYSCALE).shape
        self._data = None

    @staticmethod
    def _load_images_gt(path_to_images, path_to_gt):
        images = sorted(glob.glob(os.path.join(path_to_images, "*.png")))
        gts = sorted(glob.glob(os.path.join(path_to_gt, "*.png")))
        return images, gts

    def __len__(self):
        assert len(self._images) == len(self._gts)
        return self._size[0] * self._size[1] * len(self._images)

    def __getitem__(self, item):
        half_x = self._config.processed_x_size // 2
        half_y = self._config.processed_y_size // 2
        half_z = self._config.processed_z_size // 2

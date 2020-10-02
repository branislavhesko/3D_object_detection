from enum import Enum


class Mode(Enum):
    train = "train"
    eval = "eval"


class Config:
    batch_size = 2
    num_workers = 8
    image_shape = (512, 512)


class SegmentationConfig(Config):
    processed_z_size = 128
    processed_x_size = 128
    processed_y_size = 128

from enum import Enum


class Mode(Enum):
    train = "train"
    eval = "eval"


class Config:
    batch_size = 2
    num_workers = 8
    image_shape = (512, 512)

from enum import Enum


class Mode(Enum):
    train = "train"
    eval = "eval"


class Config:
    batch_size = 2
    num_workers = 8
    image_shape = (512, 512)
    model_path = "./data/sileane/gear/mesh.ply"
    path = "./data/sileane/gear"


class DefaultTraining:
    learning_rate = 1e-3
    weight_decay = 1e-4


class FeatureConfig:
    in_channels = 1
    out_channels = 64
    model = "ResUNetBN2E"
    normalize_features = True
    conv1_kernel_size = 5
    training = DefaultTraining()
    limit_positive_distance = 5


class SegmentationConfig(Config):
    processed_z_size = 128
    processed_x_size = 128
    processed_y_size = 128

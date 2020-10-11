import torch

from config import FeatureConfig
from utils.data_utils import find_model_opposite_points, distance


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, config: FeatureConfig, negative_loss_weight):
        self._weight = negative_loss_weight
        super(ContrastiveLoss, self).__init__()
        self._positive_loss = PositiveContrastiveLoss(config)
        self._negative_loss = NegativeContrastiveLoss(config)

    def forward(self, features_model, features_pointcloud):
        return self._positive_loss(features_model, features_pointcloud) + \
               self._weight * self._negative_loss(features_model, features_pointcloud)


class PositiveContrastiveLoss(torch.nn.Module):
    def __init__(self, config):
        super(PositiveContrastiveLoss, self).__init__()
        self._config = config

    def forward(self, features_gt, features_object):
        return distance(feats1=features_object, feats2=features_gt)


class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, config: FeatureConfig):
        super(NegativeContrastiveLoss, self).__init__()
        self._config = config

    def forward(self, features_model, features_pointcloud):
        return self._config.neg_coef - torch.relu(distance(feats1=features_model, feats2=features_pointcloud))


if __name__ == "__main__":
    neg = PositiveContrastiveLoss(FeatureConfig())
    print(find_model_opposite_points(torch.randint(low=0, high=100, size=(100, 3)).float(), torch.randint(low=0, high=100, size=(100, 3)).float(), 100)[0].shape)

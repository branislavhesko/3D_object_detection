import numpy as np
import torch

from config import FeatureConfig


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
        pass


class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, config: FeatureConfig):
        super(NegativeContrastiveLoss, self).__init__()
        self._config = config

    def forward(self, features_model, features_pointcloud):
        pass


def find_negative_matches(coords_gt, coords_object, positive_distance_limit):
    distance = torch.sqrt(torch.pow(coords_gt - coords_object, 2).sum(-1))
    return coords_gt[distance > positive_distance_limit, ...], coords_object[distance > positive_distance_limit, ...]


def find_model_opposite_points(coords_gt, coords_object, positive_distance_limit):
    distance = torch.sqrt(torch.pow(coords_gt.unsqueeze(0) - coords_gt.unsqueeze(1), 2).sum(-1))
    random_distribution = torch.distributions.exponential.Exponential(rate=5.)
    distance *= (1 - random_distribution.sample(sample_shape=distance.shape))
    furthest_points = torch.argmax(distance, dim=1)
    return coords_gt, find_positive_matches(coords_gt[furthest_points], coords_object, positive_distance_limit)[1]


def find_positive_matches(coords_gt, coords_object, positive_distance_limit):
    distance = torch.sqrt(torch.pow(coords_gt.unsqueeze(0) - coords_object.unsqueeze(1), 2).sum(-1))
    closest_points = torch.argmin(distance, dim=1)
    picked_gts = torch.arange(coords_gt.shape[0])
    correct_idx = torch.where((distance[picked_gts, closest_points] < positive_distance_limit))
    return coords_gt[picked_gts[correct_idx]], coords_object[closest_points[correct_idx]]


if __name__ == "__main__":
    neg = PositiveContrastiveLoss(FeatureConfig())
    print(find_model_opposite_points(torch.randint(low=0, high=100, size=(100, 3)).float(), torch.randint(low=0, high=100, size=(100, 3)).float(), 100)[0].shape)

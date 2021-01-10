import torch

from config import FeatureConfig
from utils.data_utils import find_model_opposite_points, distance


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, config: FeatureConfig, negative_loss_weight=1.):
        self._weight = negative_loss_weight
        super(ContrastiveLoss, self).__init__()
        self._positive_loss = PositiveContrastiveLoss(config)
        self._negative_loss = NegativeContrastiveLoss(config)

    def forward(self, features_model, features_pointcloud, positive_indices, negative_indices):
        return self._positive_loss(features_model, features_pointcloud, positive_indices) + \
               self._weight * self._negative_loss(features_model, features_pointcloud, negative_indices)


class PositiveContrastiveLoss(torch.nn.Module):
    def __init__(self, config):
        super(PositiveContrastiveLoss, self).__init__()
        self._config = config

    def forward(self, features_gt, features_object, positive_indices):
        return distance(feats1=features_object[positive_indices[:, 1], :],
                        feats2=features_gt[positive_indices[:, 0], :])


class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self, config: FeatureConfig):
        super(NegativeContrastiveLoss, self).__init__()
        self._config = config

    def forward(self, features_model, features_pointcloud, negative_indices):
        return self._config.neg_coef - torch.relu(distance(feats1=features_model[negative_indices[:, 1], :],
                                                           feats2=features_pointcloud[negative_indices[:, 0], :]))


if __name__ == "__main__":
    neg = PositiveContrastiveLoss(FeatureConfig())
    print(find_model_opposite_points(torch.randint(low=0, high=100, size=(100, 3)).float(),
                                     torch.randint(low=0, high=100, size=(100, 3)).float(), 100)[0].shape)

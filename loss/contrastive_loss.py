import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, config, negative_loss_weight):
        self._weight = negative_loss_weight
        super(ContrastiveLoss, self).__init__()
        self._positive_loss = PositiveContrastiveLoss(config)
        self._negative_loss = NegativeContrastiveLoss()

    def forward(self, features_model, features_pointcloud):
        return self._positive_loss(features_model, features_pointcloud) + \
               self._weight * self._negative_loss(features_model, features_pointcloud)


class PositiveContrastiveLoss(torch.nn.Module):
    def __init__(self, config):
        super(PositiveContrastiveLoss, self).__init__()
        self._config = config

    def forward(self, features_gt, features_object):
        pass

    def _find_closest_points(self, coords_gt, coords_object):
        distance = torch.sqrt(torch.pow(coords_gt.unsqueeze(0) - coords_object.unsqueeze(1), 2).sum(-1))
        closest_points = torch.argmin(distance, dim=1)
        picked_gts = torch.arange(coords_gt.shape[0])
        correct_idx = torch.where(distance[picked_gts, closest_points] > self._config.limit_positive_distance)
        return picked_gts[correct_idx], closest_points[correct_idx]


class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(NegativeContrastiveLoss, self).__init__()

    def forward(self, features_model, features_pointcloud):
        pass
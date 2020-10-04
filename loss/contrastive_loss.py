import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, negative_loss_weight):
        self._weight = negative_loss_weight
        super(ContrastiveLoss, self).__init__()
        self._positive_loss = PositiveContrastiveLoss()
        self._negative_loss = NegativeContrastiveLoss()

    def forward(self, features_model, features_pointcloud):
        return self._positive_loss(features_model, features_pointcloud) + \
               self._weight * self._negative_loss(features_model, features_pointcloud)


class PositiveContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(PositiveContrastiveLoss, self).__init__()


class NegativeContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super(NegativeContrastiveLoss, self).__init__()

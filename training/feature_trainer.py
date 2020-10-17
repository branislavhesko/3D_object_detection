import torch

from config import FeatureConfig
from modeling.resunet import ResUNet2, RESUNET_MODELS
from training.base_trainer import BaseTrainer


class FeatureTrainer(BaseTrainer):

    def __init__(self, config: FeatureConfig):
        super(FeatureTrainer, self).__init__()
        self._config = config
        self._model: ResUNet2 = RESUNET_MODELS[self._config.model](in_channels=self._config.in_channels,
                                                                   out_channels=self._config.out_channels,
                                                                   conv1_kernel_size=self._config.conv1_kernel_size)
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._config.training.learning_rate,
                                           weight_decay=self._config.training.weight_decay)
        self._data_loaders = {}

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self._train_single_epoch(epoch=epoch)
            if epoch % self._config.validation_frequency:
                self._validate(epoch)

    def _train_single_epoch(self, epoch):
        pass

    def _validate(self, epoch):
        pass

    def visualize(self, data):
        pass

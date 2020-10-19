import logging
import os

import torch

from config import FeatureConfig, Mode
from modeling.resunet import ResUNet2, RESUNET_MODELS
from training.base_trainer import BaseTrainer


class FeatureTrainer(BaseTrainer):

    def __init__(self, config: FeatureConfig):
        super(FeatureTrainer, self).__init__(config)
        self._model: ResUNet2 = RESUNET_MODELS[self._config.model](in_channels=self._config.in_channels,
                                                                   out_channels=self._config.out_channels,
                                                                   conv1_kernel_size=self._config.conv1_kernel_size)
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._config.training.learning_rate,
                                           weight_decay=self._config.training.weight_decay)
        self._data_loaders = {
            Mode.train: []
        }

    def save(self):
        pass

    def load(self, weights_path):
        if not os.path.exists(weights_path):
            self._logger.warning("Weights couldn't be loaded: file not found!")
            return
        state_dict = torch.load(weights_path)
        self._model.load_state_dict(state_dict["model"])
        self._optimizer.load_state_dict(state_dict["optimizer"])

    def _train_single_epoch(self, epoch):
        for idx, data in self._data_loaders[Mode.train]:
            pass

    def _validate(self, epoch):
        pass

    def visualize(self, data):
        pass


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                        level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')
    trainer = FeatureTrainer(FeatureConfig()).train(5)
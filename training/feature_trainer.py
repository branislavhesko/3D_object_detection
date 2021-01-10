import logging
import os

import MinkowskiEngine as ME
import torch

from config import FeatureConfig, Mode
from data_loaders.loader_utils import get_data_loaders, DataKeys
from loss.contrastive_loss import ContrastiveLoss
from modeling.resunet import ResUNet2, RESUNET_MODELS
from training.base_trainer import BaseTrainer


# noinspection PyCallingNonCallable
class FeatureTrainer(BaseTrainer):

    def __init__(self, config: FeatureConfig):
        super(FeatureTrainer, self).__init__(config)
        self._model: ResUNet2 = RESUNET_MODELS[self._config.model](
            in_channels=self._config.in_channels, out_channels=self._config.out_channels,
            conv1_kernel_size=self._config.conv1_kernel_size).to(self._config.device)
        self._optimizer = torch.optim.Adam(params=self._model.parameters(), lr=self._config.training.learning_rate,
                                           weight_decay=self._config.training.weight_decay)
        self._data_loaders = get_data_loaders(config=config)
        self._loss = ContrastiveLoss(self._config)

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
        for idx, data in enumerate(self._data_loaders[Mode.train]):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self._config.device)
            gt_output = self._model(ME.SparseTensor(
                coordinates=data[DataKeys.GT_COORDS],
                features=data[DataKeys.GT_FEATURES]))
            pcd_output = self._model(
                ME.SparseTensor(coordinates=data[DataKeys.PCD_COORDS], features=data[DataKeys.PCD_FEATURES]))
            loss = self._loss(gt_output.F, pcd_output.F, data[DataKeys.POS_INDICES], data[DataKeys.NEG_INDICES])
            print(loss.item())

    def _validate(self, epoch):
        pass

    def visualize(self, data):
        pass


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: [%(asctime)s] [%(name)s:%(lineno)d-%(funcName)20s()] %(message)s',
                        level=logging.INFO, datefmt='%d/%m/%Y %I:%M:%S')
    trainer = FeatureTrainer(FeatureConfig()).train(5)
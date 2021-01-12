from abc import ABCMeta, abstractmethod
import logging

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BaseTrainer(metaclass=ABCMeta):

    def __init__(self, config, *args, **kwargs):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = config
        self._writer = SummaryWriter()

    def train(self, num_epochs):
        for epoch in tqdm(range(num_epochs)):
            self._train_single_epoch(epoch=epoch)
            if epoch % self._config.validation_frequency:
                self._validate(epoch)

    @abstractmethod
    def _train_single_epoch(self, epoch):
        pass

    @abstractmethod
    def _validate(self, epoch):
        pass

    @abstractmethod
    def visualize(self, gt, pcd):
        pass

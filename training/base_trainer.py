from abc import ABCMeta, abstractmethod

from tqdm import tqdm


class BaseTrainer(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        pass

    def train(self, num_epochs):
        for epoch in tqdm(list(range(num_epochs))):
            self._train_single_epoch(epoch)

    @abstractmethod
    def _train_single_epoch(self, epoch):
        pass

    @abstractmethod
    def _validate(self, epoch):
        pass

    @abstractmethod
    def visualize(self, data):
        pass
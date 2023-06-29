from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, params):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def plot_confusion_matrix(self):
        pass

    # @abstractmethod
    # def tune(self, n_trials, hyperparameters):
    #     pass

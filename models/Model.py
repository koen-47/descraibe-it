from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def plot_confusion_matrix(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def tune(self, n_trials, n_jobs, hyperparameters):
        pass

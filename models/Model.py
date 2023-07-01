from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, x_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, x_test, y_test):
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

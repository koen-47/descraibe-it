from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

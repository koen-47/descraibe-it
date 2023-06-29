import time
from abc import ABC

import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal

from models.Model import Model


class kNN(Model):
    def __init__(self, dataset):
        self.__dataset = dataset
        self.__train = dataset.train
        self.__val = dataset.val
        self.__test = dataset.test
        self.__vectorizer = TfidfVectorizer()
        self.__vectorizer.fit_transform(self.__dataset.data["description"])
        self.__model = None
        self.__hyperparameters = {}

    def fit(self, params):
        x_train = self.__train["description"]
        y_train = self.__train["label"]
        x_train = self.__vectorizer.transform(x_train)

        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
        model.fit(x_train, y_train)

        self.__model = model

    def evaluate(self, use_val=False):
        y_true = self.__val["label"] if use_val else self.__test["label"]
        x = self.__val["description"] if use_val else self.__test["description"]
        x = self.__vectorizer.transform(x)

        y_pred = self.__model.predict(x)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        return accuracy, precision, recall, f1

    def plot_confusion_matrix(self, use_val=False, show=True, save_filepath=None):
        y_true = self.__val["label"] if use_val else self.__test["label"]
        x = self.__val["description"] if use_val else self.__test["description"]
        x = self.__vectorizer.transform(x)

        y_pred = self.__model.predict(x)
        labels = self.__dataset.decode_label(range(y_true.nunique()))
        cm_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true", display_labels=labels,
                                                             values_format=".2f", include_values=False)
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.17)
        if save_filepath is not None:
            plt.savefig(save_filepath, bbox_inches="tight")
        if show:
            plt.show()

    def __create_model(self, trial):
        n_neighbors = self.__hyperparameters["n_neighbors"]
        n_neighbors = trial.suggest_int("n_neighbors", n_neighbors["min"], n_neighbors["max"], step=n_neighbors["step"])
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        return model

    def __objective(self, trial):
        model = self.__create_model(trial)
        x_train = self.__train["description"]
        y_train = self.__train["label"]
        x_train = self.__vectorizer.transform(x_train)

        x_val = self.__val["description"]
        y_val = self.__val["label"]
        x_val = self.__vectorizer.transform(x_val)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy

    def tune(self, n_trials, hyperparameters):
        self.__hyperparameters = hyperparameters
        study = optuna.create_study(direction="maximize")
        study.optimize(self.__objective, n_trials=n_trials)
        return study.best_params

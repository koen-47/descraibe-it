import time
from abc import ABC

import numpy as np
import optuna
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal

from models.Model import Model


class SVM(Model):
    def __init__(self, dataset, params):
        self.__dataset = dataset
        self.__train = dataset.train
        self.__val = dataset.val
        self.__test = dataset.test
        self.__vectorizer = TfidfVectorizer()
        self.__vectorizer.fit_transform(self.__dataset.get_full_dataset()["description"])
        self.__model = None
        self.__params = params
        self.__param_space = {}

    def fit(self, x_train=None, y_train=None):
        x_train = self.__train["description"] if x_train is None else x_train
        y_train = self.__train["label"] if y_train is None else y_train
        x_train = self.__vectorizer.transform(x_train)

        model = SVC()
        model.fit(x_train, y_train)

        self.__model = model

    def evaluate(self, x_test=None, y_test=None, use_val=False):
        if x_test is None and y_test is None:
            y_test = self.__val["label"] if use_val else self.__test["label"]
            x_test = self.__val["description"] if use_val else self.__test["description"]

        x_test = self.__vectorizer.transform(x_test)
        y_pred = self.__model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        return accuracy, precision, recall, f1

    def cross_validate(self, n_splits):
        cv = self.__dataset.get_cv_split(n_splits=n_splits)
        total = []
        for data in cv:
            x_train = data["train"]["description"]
            y_train = data["train"]["label"]
            x_test = data["test"]["description"]
            y_test = data["test"]["label"]
            model = SVM(self.__dataset, self.__params)
            model.fit(x_train, y_train)
            results = model.evaluate(x_test, y_test)
            print(results)
            total.append(results)
        total = np.array(total)
        return np.average(total, axis=0)

    def predict(self, x):
        y_pred = self.__model.predict(x)
        return self.__dataset.decode_label(y_pred)

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
        C = self.__param_space["C"]
        gamma = self.__param_space["gamma"]
        C = trial.suggest_categorical("C", C["step"])
        gamma = trial.suggest_categorical("gamma", gamma["step"])
        model = SVC(C=C, gamma=gamma)
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
        self.__param_space = hyperparameters
        study = optuna.create_study(direction="maximize")
        study.optimize(self.__objective, n_trials=n_trials)
        return study.best_params

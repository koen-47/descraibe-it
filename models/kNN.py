import time
from abc import ABC

import numpy as np
import optuna
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal

from data.Dataset import Dataset
from models.Model import Model


class kNN(Model):
    def __init__(self, dataset, params):
        self.__dataset = dataset
        self.__train = dataset.train
        self.__val = dataset.val
        self.__test = dataset.test
        self.__vectorizer = TfidfVectorizer()
        self.__vectorizer.fit_transform(self.__dataset.get_full_dataset()["description"])
        self.__model = None
        self.__param_space = {}
        self.__params = params

    def fit(self):
        x_train = self.__vectorizer.transform(self.__train["description"])
        y_train = self.__train["label"]

        model = KNeighborsClassifier(n_neighbors=self.__params["n_neighbors"], p=self.__params["p"],
                                     weights=self.__params["weights"])
        model.fit(x_train, y_train)
        self.__model = model

    def evaluate(self, use_val=False):
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
            train_data = data["train"]
            test_data = data["test"]
            pipeline = ["make_lowercase", "expand_contractions", "clean_text"]
            dataset = Dataset(train_data=train_data, test_data=test_data, pipeline=pipeline, drop_duplicates=True)
            model = kNN(dataset, self.__params)
            model.fit()
            results = model.evaluate()
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
        n_neighbors = self.__param_space["n_neighbors"]
        weights = self.__param_space["weights"]
        p = self.__param_space["p"]

        n_neighbors = trial.suggest_int("n_neighbors", n_neighbors["min"], n_neighbors["max"], step=n_neighbors["step"])
        weights = trial.suggest_categorical("weights", weights)
        p = trial.suggest_categorical("p", p)
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
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

    def __gridsearch(self, param_space, n_jobs=1):
        train_data = self.__train
        val_data = self.__val
        train_data["i"] = -1
        val_data["i"] = 0
        data = pd.concat([train_data, val_data])
        pds = PredefinedSplit(test_fold=data["i"].tolist())
        x_train = data["description"]
        x_train = self.__vectorizer.transform(x_train)
        y_train = np.array(data["label"])
        grid = GridSearchCV(KNeighborsClassifier(), param_space, refit=True, cv=pds, n_jobs=n_jobs)
        grid.fit(x_train, y_train)
        return grid.best_params_

    def tune(self, param_space, n_trials=None, n_jobs=1, method="bayesian"):
        if method == "gridsearch" and n_trials is not None:
            raise ValueError("n_trials must be None while performing grid search.")

        self.__param_space = param_space
        if method == "bayesian":
            study = optuna.create_study(direction="maximize")
            study.optimize(self.__objective, n_trials=n_trials, n_jobs=n_jobs)
            return study.best_params
        return self.__gridsearch(param_space, n_jobs)

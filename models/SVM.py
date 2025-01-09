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
from sklearn.svm import SVC
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal

from data.Dataset import Dataset
from data.PreprocessingPipeline import PreprocessingPipeline
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

    def fit(self, x_train=None, y_train=None, x_val=None, y_val=None, use_val=False):
        if x_train is None and y_train is None:
            x_train = self.__train["description"]
            y_train = self.__train["label"]

        if x_val is None and y_val is None:
            x_val = self.__val["description"]
            y_val = self.__val["label"]

        if use_val:
            x_train = pd.concat([x_train, x_val]).reset_index(drop=True)
            y_train = pd.concat([y_train, y_val]).reset_index(drop=True)

        x_train = self.__vectorizer.transform(x_train)
        model = SVC(C=self.__params["C"], gamma=self.__params["gamma"])
        model.fit(x_train, y_train)
        self.__model = model

    def evaluate(self, x_test=None, y_test=None, use_val=False, verbose=False):
        if x_test is None and y_test is None:
            y_test = self.__val["label"] if use_val else self.__test["label"]
            x_test = self.__val["description"] if use_val else self.__test["description"]

        x_test = self.__vectorizer.transform(x_test)
        y_pred = self.__model.predict(x_test)

        accuracy = float(accuracy_score(y_test, y_pred)) * 100
        precision = float(precision_score(y_test, y_pred, average='macro')) * 100
        recall = float(recall_score(y_test, y_pred, average='macro')) * 100
        f1 = float(f1_score(y_test, y_pred, average='macro')) * 100

        if verbose:
            print(f"Results for SVM model:\n- Accuracy: {accuracy:.2f}%\n- Precision: {precision:.2f}%"
                  f"\n- Recall: {recall:.2f}%\n- F1 score: {f1:.2f}%")

        return accuracy, precision, recall, f1

    def predict(self, x, preprocessing_pipeline=None):
        if preprocessing_pipeline is not None:
            pipeline = PreprocessingPipeline(preprocessing_pipeline)
            x = pipeline.apply(x)
        x = self.__vectorizer.transform(x)
        y_pred = self.__model.predict(x)
        return self.__dataset.decode_label(y_pred)

    def cross_validate(self, n_splits, verbose=False):
        cv = self.__dataset.get_cv_split(n_splits=n_splits, as_val=True)
        best_accuracy, best_model = 0, None
        results_per_split = []
        for i, data in enumerate(cv):
            x_train = data["train"]["description"]
            y_train = data["train"]["label"]
            x_test = data["test"]["description"]
            y_test = data["test"]["label"]
            model = SVM(self.__dataset, params=self.__params)
            model.fit(x_train, y_train)
            accuracy, precision, recall, f1 = model.evaluate(x_test, y_test)
            results_per_split.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })
            if verbose:
                print(f"Accuracy on split {i+1}: {accuracy:.2f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        return best_accuracy, best_model, results_per_split

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
        plt.tight_layout()
        if save_filepath is not None:
            plt.savefig(save_filepath, bbox_inches="tight")
        if show:
            plt.show()

    def __gridsearch(self, param_space, n_jobs=1, verbose=False):
        train_data = self.__train
        val_data = self.__val
        train_data["i"] = -1
        val_data["i"] = 0
        test_fold = pd.concat([train_data, val_data])
        pds = PredefinedSplit(test_fold=test_fold["i"].tolist())

        x_train = self.__train["description"]
        x_train = self.__vectorizer.transform(x_train)
        y_train = np.array(self.__train["label"])

        grid = GridSearchCV(SVC(), param_space, refit=True, n_jobs=n_jobs, cv=pds, verbose=10 if verbose else 0)
        grid.fit(x_train, y_train)
        return grid.best_params_

    def tune(self, param_space, n_jobs=1, verbose=False):
        return self.__gridsearch(param_space, n_jobs, verbose=verbose)

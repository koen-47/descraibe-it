"""
File containing all functionality related to the kNN model.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from models.Model import Model
from data.PreprocessingPipeline import PreprocessingPipeline


class kNN(Model):
    """
    Class to implement functionality for the kNN model.
    """
    def __init__(self, dataset, params):
        """
        Constructor for the kNN model.
        :param dataset: Dataset instance containing the whole dataset to run the kNN on.
        :param params: dictionary determining the parameters of the kNN model.
        """

        # Parameters for the dataset.
        self.__dataset = dataset
        self.__train = dataset.train
        self.__val = dataset.val
        self.__test = dataset.test

        # Create vectorizer and fit it to the training set of the dataset.
        self.__vectorizer = TfidfVectorizer()
        self.__vectorizer.fit_transform(self.__dataset.get_full_dataset()["description"])

        # Parameters for the model.
        self.__model = None
        self.__param_space = {}
        self.__params = params

    def fit(self, x_train=None, y_train=None, x_val=None, y_val=None, use_val=False):
        """
        Fits the kNN to a (specified) dataset.
        :param x_train: features of the training set. If not specified, it will default to the training features of the
        dataset specified during instantiation.
        :param y_train: labels of the training set. If not specified, it will default to the training features of the
        dataset specified during instantiation.
        :param x_val: features of the validation set. If not specified, it will default to the validation features of
        the dataset specified during instantiation.
        :param y_val: labels of the validation set. If not specified, it will default to the validation features of
        the dataset specified during instantiation.
        :param use_val: flag to denote if the training data to fit the model to will also include the validation set.
        """

        # Get the training data from self.dataset if it is not explicitly specified.
        if x_train is None and y_train is None:
            x_train = self.__train["description"]
            y_train = self.__train["label"]

        # Get the validation data from self.dataset if it is not explicitly specified.
        if x_val is None and y_val is None:
            x_val = self.__val["description"]
            y_val = self.__val["label"]

        # Concatenate training and validation data is use_val is set.
        if use_val:
            x_train = pd.concat([x_train, x_val]).reset_index(drop=True)
            y_train = pd.concat([y_train, y_val]).reset_index(drop=True)

        # Convert training data using TF-IDF vectorizer.
        x_train = self.__vectorizer.transform(x_train)

        # Fit the model to training data.
        model = KNeighborsClassifier(n_neighbors=self.__params["n_neighbors"], p=self.__params["p"],
                                     weights=self.__params["weights"])
        model.fit(x_train, y_train)
        self.__model = model

    def evaluate(self, x_test=None, y_test=None, use_val=False, verbose=False):
        """
        Evaluate the fitted model on the test or validation set.
        :param x_test: specified test/validation features to evaluate on.
        :param y_test: specified test/validation labels to evaluate on.
        :param use_val: flag to indicate if the model should be evaluated on the test set or validation set.
        :param verbose: flag to indicate if the results should be printed at the end.
        :return: quadruple consisting of the accuracy, precision, recall and f1-score.
        """

        # Get either the validation or test set if they are specified.
        if x_test is None and y_test is None:
            y_test = self.__val["label"] if use_val else self.__test["label"]
            x_test = self.__val["description"] if use_val else self.__test["description"]

        # Convert test/validation data using TF-IDF vectorizer and perform prediction.
        x_test = self.__vectorizer.transform(x_test)
        y_pred = self.__model.predict(x_test)

        # Compute evaluation metrics.
        accuracy = float(accuracy_score(y_test, y_pred)) * 100
        precision = float(precision_score(y_test, y_pred, average='macro')) * 100
        recall = float(recall_score(y_test, y_pred, average='macro')) * 100
        f1 = float(f1_score(y_test, y_pred, average='macro')) * 100

        # Print metric results if verbosity is set.
        if verbose:
            print(f"Results for kNN model:\n- Accuracy: {accuracy:.2f}%\n- Precision: {precision:.2f}%"
                  f"\n- Recall: {recall:.2f}%\n- F1 score: {f1:.2f}%")

        return accuracy, precision, recall, f1

    def predict(self, x, preprocessing_pipeline=None):
        """
        Perform a prediction on the specified data points.
        :param x: specified data point to perform the prediction on.
        :param preprocessing_pipeline: order of preprocessing steps to perform (see PreprocessingPipeline for clarification).
        :return: the predicted label (as a string).
        """

        # Preprocess the specified data points (if necessary).
        if preprocessing_pipeline is not None:
            pipeline = PreprocessingPipeline(preprocessing_pipeline)
            x = pipeline.apply(x)
        x = self.__vectorizer.transform(x)

        # Perform prediction.
        y_pred = self.__model.predict(x)
        return self.__dataset.decode_label(y_pred)

    def cross_validate(self, n_splits, verbose=False):
        """
        Perform cross validation on self.dataset based on the specified number of splits to compute the best performing
        model.
        :param n_splits: specified number of splits to perform cross validation on.
        :param verbose: flag to denote if results per split are printed.
        :return: triple consisting of the best accuracy of the best performing model, the best performing model,
        and the results per split (accuracy, precision, recall, f1-score, train + validation loss/accuracy per epoch).
        """

        # Concatenate the training + validation sets and split them into the specified number of splits.
        cv = self.__dataset.get_cv_split(n_splits=n_splits, as_val=True)
        best_accuracy, best_model = 0, None
        results_per_split = []

        # Iterate over each split.
        for i, data in enumerate(cv):
            # Get training + validation set of that split.
            x_train = data["train"]["description"]
            y_train = data["train"]["label"]
            x_test = data["test"]["description"]
            y_test = data["test"]["label"]

            # Fit the model to training set and evaluate on the validation set.
            model = kNN(self.__dataset, params=self.__params)
            model.fit(x_train, y_train)
            accuracy, precision, recall, f1 = model.evaluate(x_test, y_test)

            # Record results for that split.
            results_per_split.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

            # Print results if verbosity is set.
            if verbose:
                print(f"Accuracy on split {i+1}: {accuracy:.2f}")

            # Record best performing model and accuracy of that model.
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        return best_accuracy, best_model, results_per_split

    def __gridsearch(self, param_space, n_jobs=1, verbose=False):
        """
        Performs grid search on the specified parameter space.
        :param param_space: parameter space to search through.
        :param n_jobs: number of processes to perform a search on.
        :param verbose: flag to denote if the results per searched parameter will be printed.
        :return: best performing parameters.
        """

        # Set up training + validation data to perform gridsearch on.
        train_data = self.__train
        val_data = self.__val
        train_data["test_fold"] = -1
        val_data["test_fold"] = 0

        # Compute the split based on the training + validation data.
        data = pd.concat([train_data, val_data])
        pds = PredefinedSplit(test_fold=data["test_fold"].tolist())

        # Convert the data with TF-IDF vectorizer.
        x_data = data["description"]
        x_data = self.__vectorizer.transform(x_data)
        y_data = np.array(data["label"])

        # Perform grid search.
        grid = GridSearchCV(KNeighborsClassifier(), param_space, refit=True, n_jobs=n_jobs, cv=pds,
                            verbose=10 if verbose else 0)
        grid.fit(x_data, y_data)
        return grid.best_params_

    def tune(self, param_space, n_jobs=1, verbose=False):
        """
        Performs grid search on the specified parameter space.
        :param param_space: parameter space to search through.
        :param n_jobs: number of processes to perform a search on.
        :param verbose: flag to denote if the results per searched parameter will be printed.
        :return: best performing parameters.
        """
        tuned_params = self.__gridsearch(param_space, n_jobs, verbose=verbose)
        self.__model = kNN(self.__dataset, tuned_params)
        return tuned_params

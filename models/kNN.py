import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from models.Model import Model
from data.PreprocessingPipeline import PreprocessingPipeline


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

    def fit(self, use_val=False):
        x_train = self.__train["description"]
        y_train = self.__train["label"]

        if use_val:
            x_val = self.__val["description"]
            y_val = self.__val["label"]
            x_train = pd.concat([x_train, x_val]).reset_index(drop=True)
            y_train = pd.concat([y_train, y_val]).reset_index(drop=True)

        x_train = self.__vectorizer.transform(x_train)
        model = KNeighborsClassifier(n_neighbors=self.__params["n_neighbors"], p=self.__params["p"],
                                     weights=self.__params["weights"])
        model.fit(x_train, y_train)
        self.__model = model

    def evaluate(self, use_val=False, verbose=False):
        x_test = self.__val["description"] if use_val else self.__test["description"]
        x_test = self.__vectorizer.transform(x_test)

        y_test = self.__val["label"] if use_val else self.__test["label"]
        y_pred = self.__model.predict(x_test)

        accuracy = float(accuracy_score(y_test, y_pred)) * 100
        precision = float(precision_score(y_test, y_pred, average='macro')) * 100
        recall = float(recall_score(y_test, y_pred, average='macro')) * 100
        f1 = float(f1_score(y_test, y_pred, average='macro')) * 100

        if verbose:
            print(f"Results for kNN model:\n- Accuracy: {accuracy:.2f}%\n- Precision: {precision:.2f}%"
                  f"\n- Recall: {recall:.2f}%\n- F1 score: {f1:.2f}%")

        return accuracy, precision, recall, f1

    def predict(self, x, preprocessing_pipeline=None):
        if preprocessing_pipeline is not None:
            pipeline = PreprocessingPipeline(preprocessing_pipeline)
            x = pipeline.apply(x)
        x = self.__vectorizer.transform(x)
        y_pred = self.__model.predict(x)
        return self.__dataset.decode_label(y_pred)

    def plot_confusion_matrix(self, use_val=False, show=True, save_filepath=None, dark_mode=True):
        y_true = self.__val["label"] if use_val else self.__test["label"]
        x = self.__val["description"] if use_val else self.__test["description"]
        x = self.__vectorizer.transform(x)

        y_pred = self.__model.predict(x)
        labels = self.__dataset.decode_label(range(y_true.nunique()))

        color = "#0d1117" if not dark_mode else "#F0F6FC"
        not_color = "#0d1117" if dark_mode else "#F0F6FC"

        cm = confusion_matrix(y_true, y_pred, normalize="true")
        colors = ["#ff4747", color, "#459abd"]
        custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=custom_cmap, values_format=".2f", include_values=False)

        ax = disp.ax_
        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_color(color)
        ax.spines["top"].set_color(color)
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="x", colors=color)
        ax.tick_params(axis="y", colors=color)

        colorbar = disp.im_.colorbar
        colorbar.ax.yaxis.set_tick_params(color=color)
        colorbar.ax.yaxis.label.set_color(color)
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_color(color)

        im = disp.im_
        im.axes.grid(which="minor", color=not_color, linestyle="-", linewidth=2)
        im.axes.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
        im.axes.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
        im.axes.grid(which="minor", color=not_color, linestyle="-", linewidth=1.5)
        im.axes.tick_params(which="minor", bottom=False, left=False)

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
        train_data["test_fold"] = -1
        val_data["test_fold"] = 0

        data = pd.concat([train_data, val_data])
        pds = PredefinedSplit(test_fold=data["test_fold"].tolist())

        x_data = data["description"]
        x_data = self.__vectorizer.transform(x_data)
        y_data = np.array(data["label"])

        grid = GridSearchCV(KNeighborsClassifier(), param_space, refit=True, n_jobs=n_jobs, cv=pds,
                            verbose=10 if verbose else 0)
        grid.fit(x_data, y_data)
        return grid.best_params_

    def tune(self, param_space, n_jobs=1, verbose=False):
        return self.__gridsearch(param_space, n_jobs, verbose=verbose)

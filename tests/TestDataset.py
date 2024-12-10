import os
import unittest

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

from data.Dataset import Dataset
from data.PreprocessingPipeline import PreprocessingPipeline


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"
        # self.pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "lemmatize", "clean_text"]
        # self.dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.3, val_split=0.2,
        #                        shuffle=True, pipeline=self.pipeline, drop_duplicates=True)

    def test_dataset_individual_datapoint(self):
        print(self.dataset)
        x = "3. I simply can't with this right now."
        preprocessing = PreprocessingPipeline(self.pipeline)
        print(preprocessing.apply(x))

    def test_dataset_set_data(self):
        train_data = self.dataset.train[:1000]
        val_data = self.dataset.val[:1000]
        test_data = self.dataset.test[:1000]
        dataset = Dataset(train_data=train_data, val_data=val_data, test_data=test_data, preprocess=self.pipeline,
                          encode_labels=False, drop_duplicates=True)

        print(dataset.val["label"].value_counts())
        print(val_data["label"].value_counts())

    def test_dataset_definition_errors(self):
        train_data = self.dataset.train[:1000]
        val_data = self.dataset.val[:1000]
        test_data = self.dataset.test[:1000]
        dataset = Dataset(csv_path="", train_data=train_data, val_data=val_data, test_data=test_data,
                          preprocess=self.pipeline, encode_labels=False, drop_duplicates=True)

    def test_cv_split(self):
        pipeline = ["make_lowercase", "expand_contractions" "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, preprocess=pipeline, drop_duplicates=True)
        cv = dataset.get_cv_split(n_splits=4)
        print(cv)
        for split in cv:
            print(len(split["train"]))
            print(len(split["test"]))

    def test_create_splits(self):
        import re
        pipeline = []
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.3, val_split=0.2,
                          shuffle=True, preprocess=pipeline, drop_duplicates=True, encode_labels=False)

        len_train = len(dataset.train)
        len_val = len(dataset.val)
        len_test = len(dataset.test)

        for i, desc in enumerate(dataset.train["description"]):
            split_desc = desc.split()
            if bool(re.search(r"\b\d+\.", split_desc[0])) or bool(re.search(r"\b\d+\)", split_desc[0])):
                dataset.train.loc[i, "description"] = " ".join(split_desc[1:])
            # else:
            #     print(desc)
        dataset.train.to_csv("../data/splits/train.csv", index=False)

        for i, desc in enumerate(dataset.test["description"]):
            split_desc = desc.split()
            if bool(re.search(r"\b\d+\.", split_desc[0])) or bool(re.search(r"\b\d+\)", split_desc[0])):
                dataset.test.loc[i, "description"] = " ".join(split_desc[1:])
            # else:
            #     print(desc)
        dataset.test.to_csv("../data/splits/test.csv", index=False)

        for i, desc in enumerate(dataset.val["description"]):
            split_desc = desc.split()
            if bool(re.search(r"\b\d+\.", split_desc[0])) or bool(re.search(r"\b\d+\)", split_desc[0])):
                dataset.val.loc[i, "description"] = " ".join(split_desc[1:])
            # else:
            #     print(desc)
        dataset.val.to_csv("../data/splits/val.csv", index=False)

        print(len_train)
        print(len_val)
        print(len_test)
        print(len_train + len_val + len_test)

    def test_n_labels(self):
        train_data = pd.read_csv("../data/splits/train.csv")
        test_data = pd.read_csv("../data/splits/test.csv")
        val_data = pd.read_csv("../data/splits/val.csv")
        pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
        self.dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline,
                               encode_labels=True, drop_duplicates=True, shuffle=True)
        print(self.dataset.train["label"].nunique())

        n_total_data = len(train_data) + len(test_data) + len(val_data)
        print(len(train_data) / n_total_data, len(train_data))
        print(len(test_data) / n_total_data, len(test_data))
        print(len(val_data) / n_total_data, len(val_data))
        print(n_total_data)

    def test_visualize_class_balance(self):
        train_data = pd.read_csv("../data/splits/train.csv")
        test_data = pd.read_csv("../data/splits/test.csv")
        val_data = pd.read_csv("../data/splits/val.csv")
        pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
        self.dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline,
                               encode_labels=False)

        self.visualize_class_balance(self.dataset.train, self.dataset.test, self.dataset.val)

    def visualize_class_balance(self, train, test, val=None, dark_mode=True):
        train_count = train["label"].value_counts().sort_index()
        train_count = pd.DataFrame({"Word": train_count.index, "Frequency": train_count.values})
        test_count = test["label"].value_counts().sort_index()
        test_count = pd.DataFrame({"Word": test_count.index, "Frequency": test_count.values})
        val_count = val["label"].value_counts().sort_index()
        val_count = pd.DataFrame({"Word": val_count.index, "Frequency": val_count.values})

        data = pd.DataFrame({
            "Word": train_count["Word"],
            "Train set": train_count["Frequency"],
            "Test set": test_count["Frequency"],
            "Validation set": val_count["Frequency"]
        })

        fig = plt.figure(figsize=(12, 8))
        sns.set(font_scale=1.1)
        ax = data.set_index("Word").plot(kind="bar", stacked=True, color=["#ff4747", "#459abd", "#43b97f"], alpha=0.75)

        for bars in ax.containers:
            for bar in bars:
                bar.set_edgecolor("#0d1117")
                bar.set_linewidth(1.5)

        color = "#0d1117" if not dark_mode else "#F0F6FC"
        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="x", colors=color)
        ax.tick_params(axis="y", colors=color)
        ax.grid(False)

        plt.ylim(0, 7500)
        plt.yticks(np.arange(0, 7500, 1750))
        plt.ylabel("Frequency")
        plt.legend(frameon=False, labelcolor=color, ncol=3, bbox_to_anchor=(0.5, 1.02), loc="lower center")
        plt.tight_layout()
        plt.savefig(f"../data/resources/class_balance_chart_{'dark' if dark_mode else 'light'}.png",
                    transparent=True)

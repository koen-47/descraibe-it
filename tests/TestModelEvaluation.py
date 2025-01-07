import json
import os
import unittest

import warnings

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.simplefilter("default")
warnings.filterwarnings(
    "ignore",
    message=r"non-integer arguments to randrange\(\).*deprecated since Python 3\.10.*",
    category=DeprecationWarning
)

import pandas as pd
import matplotlib.pyplot as plt

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.LSTM import LSTM
from models.kNN import kNN
from models.SVM import SVM

import warnings


class TestModelEvaluation(unittest.TestCase):
    def test_evaluate_lstm(self):
        train_data = pd.read_csv("../data/splits/train.csv")
        test_data = pd.read_csv("../data/splits/test.csv")
        val_data = pd.read_csv("../data/splits/val.csv")
        pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
        self.dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline)

        glove = GloVeEmbedding(f"../data/embeddings/glove.840B.300d.txt", dimensionality=300)
        # glove = GloVeEmbedding(f"../data/embeddings/glove.6B.100d.txt", dimensionality=100)

        params = {
            "lstm_layers": [{"bidirectional": True, "units": 448}],
            "fc_layers": [{"units": 384, "dropout_p": 0.7}],
            "early_stopping": {"patience": 20},
            "scheduler": {"initial_learning_rate": 0.0001, "decay_steps": 250},
            "optimizer": {"name": "adam", "beta_1": 0.906, "beta_2": 0.955},
            "misc": {"epochs": 500, "batch_size": 256, "save_filepath": "./models/saved/lstm.h5", "verbose": 1}
        }

        model = LSTM(self.dataset, embedding=glove, params=params)
        best_accuracy, best_model, results_per_split = model.cross_validate(5, verbose=True)
        print(f"Best accuracy: {best_accuracy}")
        accuracy, precision, recall, f1 = best_model.evaluate(verbose=True)
        results_per_split.append({
            "best_accuracy": accuracy,
            "best_precision": precision,
            "best_recall": recall,
            "best_f1": f1
        })

        with open("../results/lstm/lstm_evaluation.json", "w") as file:
            json.dump(results_per_split, file, indent=3)

    def test_generate_val_loss_plots(self):
        with open("../results/lstm/lstm_evaluation.json", "r") as file:
            results = json.load(file)

        train_losses = [result["train_loss_history"] for result in results[:-1]]
        val_losses = [result["val_loss_history"] for result in results[:-1]]
        min_x = min(min([len(loss) for loss in train_losses]), min([len(loss) for loss in val_losses]))
        x = list(range(1, min_x + 1))

        train_losses = np.array([loss[:min_x] for loss in train_losses])
        val_losses = np.array([loss[:min_x] for loss in val_losses])

        train_mu = train_losses.mean(axis=0)
        train_sigma = train_losses.std(axis=0)
        val_mu = val_losses.mean(axis=0)
        val_sigma = val_losses.std(axis=0)

        dark_mode = False
        color = "#0d1117" if not dark_mode else "#F0F6FC"
        not_color = "#0d1117" if dark_mode else "#F0F6FC"

        fig, ax = plt.subplots(1)
        ax.plot(x, train_mu, label="Validation loss", color="#459abd")
        ax.fill_between(x, train_mu + train_sigma, train_mu - train_sigma, facecolor="#459abd", alpha=0.5)
        ax.plot(x, val_mu, label="Validation loss", color="#ff4747")
        ax.fill_between(x, val_mu + val_sigma, val_mu - val_sigma, facecolor="#ff4747", alpha=0.5)

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

        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy Loss")

        plt.tight_layout()
        plt.show()

    def test_evaluate_svm(self):
        train_data = pd.read_csv("../data/splits/train.csv")
        test_data = pd.read_csv("../data/splits/test.csv")
        val_data = pd.read_csv("../data/splits/val.csv")
        pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
        self.dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline)

        with open("./results/svm/svm_results.json") as file:
            svm_results = json.load(file)
        best_params = svm_results["tuning"]["best_params"]
        model = SVM(self.dataset, best_params)

        best_accuracy, best_model, results_per_split = model.cross_validate(5, verbose=True)
        print(f"Best accuracy: {best_accuracy}")
        accuracy, precision, recall, f1 = best_model.evaluate(verbose=True)
        results_per_split.append({
            "best_accuracy": accuracy,
            "best_precision": precision,
            "best_recall": recall,
            "best_f1": f1
        })

        with open("../results/svm/svm_evaluation.json", "w") as file:
            json.dump(results_per_split, file, indent=3)

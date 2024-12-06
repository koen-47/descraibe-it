import json
import unittest

import pandas as pd

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.LSTM import LSTM
from models.kNN import kNN
from models.SVM import SVM


class TestModelTuning(unittest.TestCase):
    def setUp(self):
        train_data = pd.read_csv("../data/splits/train.csv")
        test_data = pd.read_csv("../data/splits/test.csv")
        val_data = pd.read_csv("../data/splits/val.csv")
        pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
        self.dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline)

    def test_tune_knn(self):
        model = kNN(self.dataset, {})
        param_space = {
            "n_neighbors": list(range(1, 51)),
            "weights": ["distance", "uniform"],
            "p": [2]
        }

        best_params = model.tune(param_space, n_jobs=None, verbose=True)
        model = kNN(self.dataset, best_params)
        model.fit(use_val=True)

        accuracy, precision, recall, f1_score = model.evaluate()
        knn_results = {
            "performance": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "tuning": {
                "param_space": param_space,
                "best_params": best_params
            }
        }

        with open("../results/knn/knn_results.json", "w") as file:
            json.dump(knn_results, file, indent=3)
        model.plot_confusion_matrix(show=False, save_filepath="../results/knn/knn_confusion_matrix.png")

    def test_tune_svm(self):
        model = SVM(self.dataset, {})
        param_space = {
            "C": [0.1, 1., 10., 100., 1000.],
            "gamma": [0.0001, 0.001, 0.01, 0.1, 1., 10.],
        }

        best_params = model.tune(n_jobs=None, param_space=param_space)
        model = SVM(self.dataset, best_params)
        model.fit()

        accuracy, precision, recall, f1_score = model.evaluate()
        svm_results = {
            "performance": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "tuning": {
                "param_space": param_space,
                "best_params": best_params
            }
        }

        with open("../results/svm/svm_results.json", "w") as file:
            json.dump(svm_results, file, indent=3)
        model.plot_confusion_matrix(show=False, save_filepath="../results/svm/svm_confusion_matrix.png")

    def test_tune_lstm(self):
        # glove = GloVeEmbedding(f"../data/embeddings/glove.6B.100d.txt", dimensionality=100)
        glove = GloVeEmbedding(f"../data/embeddings/glove.840B.300d.txt", dimensionality=300)
        model = LSTM(self.dataset, glove, {})

        param_space = {
            "architecture": {
                "n_lstm_units": {"min": 64, "max": 512, "step": 64},
                "n_fc_layers": {"min": 1, "max": 2, "step": 1},
                "n_fc_units": {"min": 128, "max": 1024, "step": 128},
                "dropout_p": {"min": 0.1, "max": 0.7, "step": 0.2}
            },
            "optimizer": {
                # "scheduler": {
                #     "initial_lr": {"min": 1e-4, "max": 1e-3},
                #     "decay_steps": {"min": 25, "max": 250, "step": 25},
                # },
                "adam": {
                    "beta_1": {"min": 0.9, "max": 0.9},
                    "beta_2": {"min": 0.999, "max": 0.999},
                },
                # "sgd": {
                #     "momentum": {"min": 0., "max": 0.999},
                # }
            }
        }

        model.tune(n_trials=50, param_space=param_space, save=f"../results/lstm/lstm_results_arch_1.json")

    def test_tf_gpu(self):
        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

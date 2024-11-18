import json
import os.path
import unittest

import pandas as pd

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.LSTM import LSTM
from models.kNN import kNN
from models.SVM import SVM


class TestModelExperimentation(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"
        self.embedding_filepath = f"{os.path.dirname(__file__)}/../data/embeddings"

    def __get_datasets(self):
        pipeline1 = ["make_lowercase", "expand_contractions", "clean_text"]
        pipeline2 = ["make_lowercase", "expand_contractions" "clean_text", "remove_stopwords"]
        pipeline3 = ["make_lowercase", "expand_contractions", "clean_text", "remove_stopwords", "lemmatize"]
        dataset1 = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=pipeline1, drop_duplicates=True)
        dataset2 = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=pipeline2, drop_duplicates=True)
        dataset3 = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=pipeline3, drop_duplicates=True)
        return dataset1, dataset2, dataset3

    def test_fit_lstm(self):
        pipeline = ["make_lowercase", "expand_contractions", "clean_text"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.840B.300d.txt", dimensionality=300)

        params = {
            "lstm_layers": [{"bidirectional": True, "units": 480}],
            "fc_layers": [{"units": 656, "dropout_p": 0.7}],
            "early_stopping": {"patience": 10, "verbose": 1},
            "scheduler": {"initial_learning_rate": 0.001, "decay_steps": 50},
            "misc": {"epochs": 10000, "lr": 0.0085, "batch_size": 256, "save_filepath": "./models/saved/lstm.h5"}
        }

        model = LSTM(dataset, embedding=glove, params=params)
        model.fit()
        accuracy, precision, recall, f1, incorrect_df = model.evaluate(return_incorrect=True)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")
        print(f"Misclassification ratio: {len(incorrect_df) / len(dataset.test)}")

    def test_cross_validate_lstm(self):
        pipeline = ["make_lowercase", "expand_contractions", "clean_text"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.6B.100d.txt", dimensionality=100)

        params = {
            "lstm_layers": [{"bidirectional": True, "units": 480}],
            "fc_layers": [{"units": 656, "dropout_p": 0.7}],
            "early_stopping": {"patience": 10, "verbose": 1},
            "scheduler": {"initial_learning_rate": 0.001, "decay_steps": 50},
            "misc": {"epochs": 25, "lr": 0.0085, "batch_size": 512, "save_filepath": "./models/saved/lstm.h5"}
        }

        model = LSTM(dataset, embedding=glove, params=params)
        scores, incorrect_df = model.cross_validate(n_splits=10, return_incorrect=True)
        incorrect_df.to_csv("./incorrect_df_test.csv", index_label=False, index=False)

    def test_tune_lstm(self):
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.6B.100d.txt", dimensionality=100)
        dataset1 = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=["make_lowercase", "clean_text"], drop_duplicates=True)

        model1 = LSTM(dataset1, embedding=glove)
        hyperparameters = {
            "n_lstm_units": {"min": 16, "max": 512, "step": 32},
            "n_fc_layers": {"min": 1, "max": 5, "step": 1},
            "n_fc_units": {"min": 16, "max": 2048, "step": 64},
            "dropout_p": {"min": 0., "max": 0.9, "step": 0.1}
        }
        model1.tune(n_trials=25, hyperparameters=hyperparameters)

    def test_fit_knn(self):
        print("adfasdf")
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        params = {"n_neighbors": 16, "weights": "distance", "p": 2}
        model = kNN(dataset, params)
        model.fit()
        print(model.evaluate())
        model.plot_confusion_matrix(show=False, save_filepath="../visualizations/confusion_matrix_knn.png")

    def test_fit_knn_small(self):
        data = pd.read_csv("../data/saved/descriptions_25.csv").sample(frac=1).reset_index(drop=True)
        train_data = data[:12000]
        val_data = data[12000:15000]
        test_data = data[15000:20000]
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(train_data=train_data, val_data=val_data, test_data=test_data, pipeline=pipeline,
                          drop_duplicates=True)
        params = {"n_neighbors": 16, "weights": "distance", "p": 2}
        model = kNN(dataset, params)
        model.fit()
        print(model.evaluate(use_val=True))
        # model.plot_confusion_matrix(show=False, save_filepath="../visualizations/confusion_matrix_knn.png")

    def test_cross_validate_knn(self):
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        params = {"n_neighbors": 8}
        model = kNN(dataset, params)
        model.cross_validate(n_splits=3)

    def test_tune_knn(self):
        pipeline = ["make_lowercase", "expand_contractions", "clean_text", "remove_stopwords", "lemmatize"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                          val_split=0.2, shuffle=True, pipeline=pipeline, drop_duplicates=True)
        model = kNN(dataset, {})
        param_space = {
            "n_neighbors": list(range(1, 51)),
            "weights": ["distance", "uniform"],
            "p": [2]
        }
        best_params = model.tune(param_space, method="gridsearch", n_jobs=None)
        model = kNN(dataset, best_params)
        model.fit()

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

    def test_svm(self):
        pipeline = ["make_lowercase", "clean_text"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        params = {}
        model = SVM(dataset, params)
        model.fit()
        print(model.evaluate())
        model.plot_confusion_matrix(save_filepath="../visualizations/confusion_matrix_svm.png")

    def test_tune_svm(self):
        pipeline = ["make_lowercase", "expand_contractions", "clean_text", "remove_stopwords", "lemmatize"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        params = {}
        model = SVM(dataset, params)
        param_space = {
            "C": [0.1, 1., 10., 100., 1000.],
            "gamma": [0.0001, 0.001, 0.01, 0.1, 1., 10.],
        }

        best_params = model.tune(n_jobs=None, param_space=param_space, method="gridsearch")

        print("finished tuning")

        model = SVM(dataset, best_params)
        model.fit()
        print("finished fitting")

        accuracy, precision, recall, f1_score = model.evaluate()
        print("finished evaluating")
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

    def test_cross_validate_svm(self):
        pipeline = ["make_lowercase", "expand_contractions" "clean_text"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        params = {"C": 10., "gamma": 0.1}
        model = SVM(dataset, params)
        model.cross_validate(n_splits=3)


class TestMultipleDatasetExperimentation(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"
        self.embedding_filepath = f"{os.path.dirname(__file__)}/../data/embeddings"

    def __get_datasets(self):
        pipeline1 = ["make_lowercase", "expand_contractions", "clean_text"]
        pipeline2 = ["make_lowercase", "expand_contractions", "clean_text", "remove_stopwords"]
        pipeline3 = ["make_lowercase", "expand_contractions", "clean_text", "remove_stopwords", "lemmatize"]
        dataset1 = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=pipeline1, drop_duplicates=True)
        dataset2 = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=pipeline2, drop_duplicates=True)
        dataset3 = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=pipeline3, drop_duplicates=True)
        return dataset1, dataset2, dataset3

    def test_tune_knn(self):
        datasets = self.__get_datasets()
        param_space = {
            "n_neighbors": range(1, 101),
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }

        for i, dataset in enumerate(datasets):
            model = kNN(dataset, {})
            best_params = model.tune(param_space, n_jobs=6, method="gridsearch")
            model = kNN(dataset, best_params)
            model.fit()
            print(f"Dataset {i + 1}")
            print(f"Best parameters: {best_params}")
            print(f"Results: {model.evaluate()}")

    def test_tune_svm(self):
        datasets = self.__get_datasets()
        param_space = {
            "C": {"min": 0.1, "max": 100, "step": [0.1, 1., 10., 100., 1000.]},
            "gamma": {"min": 0.01, "max": 10, "step": [0.0001, 0.001, 0.01, 0.1, 1., 10.]},
        }

        for dataset in datasets:
            model = SVM(dataset, {})
            best_params = model.tune(n_trials=10, n_jobs=-1, param_space=param_space, method="gridsearch")
            model = SVM(dataset, best_params)
            model.fit()
            print(model.evaluate())

import os.path
import unittest

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.LSTM import LSTM
from models.kNN import kNN


class TestModelExperimentation(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"
        self.embedding_filepath = f"{os.path.dirname(__file__)}/../data/embeddings"

    def test_fit_lstm(self):
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.6B.100d.txt", dimensionality=100)

        params = {
            "lstm_layers": [{"bidirectional": True, "units": 480}],
            "fc_layers": [{"units": 656, "dropout_p": 0.7}],
            "early_stopping": {"patience": 10, "verbose": 1},
            "scheduler": {"initial_learning_rate": 0.001, "decay_steps": 50},
            "misc": {"epochs": 1, "lr": 0.0085, "batch_size": 64, "save_filepath": "./models/saved/lstm.h5"}
        }

        model = LSTM(dataset, embedding=glove)
        model.fit(params)
        accuracy, precision, recall, f1, incorrect_df = model.evaluate()

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 score: {f1}")
        print(f"Misclassification ratio: {len(incorrect_df) / len(dataset.test)}")

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
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        model = kNN(dataset)
        model.fit(params={})
        model.evaluate()
        model.plot_confusion_matrix(save_filepath="../visualizations/confusion_matrix_knn.png")

    def test_tune_knn(self):
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        model = kNN(dataset)
        hyperparameters = {
            "n_neighbors": {"min": 1, "max": 50, "step": 1},
        }

        best_params = model.tune(n_trials=10, hyperparameters=hyperparameters)
        print(best_params)

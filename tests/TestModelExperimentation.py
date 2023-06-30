import os.path
import unittest

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
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.6B.100d.txt", dimensionality=100)

        params = {
            "lstm_layers": [{"bidirectional": True, "units": 480}],
            "fc_layers": [{"units": 656, "dropout_p": 0.7}],
            "early_stopping": {"patience": 10, "verbose": 1},
            "scheduler": {"initial_learning_rate": 0.001, "decay_steps": 50},
            "misc": {"epochs": 1, "lr": 0.0085, "batch_size": 128, "save_filepath": "./models/saved/lstm.h5"}
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
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.6B.100d.txt", dimensionality=100)

        params = {
            "lstm_layers": [{"bidirectional": True, "units": 480}],
            "fc_layers": [{"units": 656, "dropout_p": 0.7}],
            "early_stopping": {"patience": 10, "verbose": 1},
            "scheduler": {"initial_learning_rate": 0.001, "decay_steps": 50},
            "misc": {"epochs": 1, "lr": 0.0085, "batch_size": 128, "save_filepath": "./models/saved/lstm.h5"}
        }

        model = LSTM(dataset, embedding=glove, params=params)
        scores, incorrect_df = model.cross_validate(n_splits=2, return_incorrect=True)
        incorrect_df.to_csv("./incorrect_df_test.csv", index_label=False, index=False)

        # print(len(incorrect_df) / len(dataset.get_full_dataset()))
        # print(scores[0])
        # print(1 - scores[0])

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
        model.fit(params={"n_neighbors": 8})
        print(model.evaluate())
        model.plot_confusion_matrix(show=False, save_filepath="../visualizations/confusion_matrix_knn.png")

    def test_cross_validate_knn(self):
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        model = kNN(dataset)
        model.cross_validate(params={"n_neighbors": 8}, n_splits=3)

    def test_tune_knn(self):
        hyperparameters = {
            "n_neighbors": {"min": 1, "max": 50, "step": 1},
            "weights": ["uniform", "distance"],
            "p": [1, 2]
        }

        optimal = []
        for dataset in self.__get_datasets():
            model = kNN(dataset)
            optimal_params = model.tune(n_trials=2, hyperparameters=hyperparameters)
            optimal.append(optimal_params)
        print(optimal)

    def test_svm(self):
        pipeline = ["make_lowercase", "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        model = SVM(dataset)
        model.fit(params={})
        model.evaluate()
        model.plot_confusion_matrix(save_filepath="../visualizations/confusion_matrix_svm.png")

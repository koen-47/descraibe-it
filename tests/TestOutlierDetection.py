import os
import unittest

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.LSTM import LSTM


class TestOutlierDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"
        self.embedding_filepath = f"{os.path.dirname(__file__)}/../data/embeddings"

    def test_model_outlier_detection(self):
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.6B.100d.txt", dimensionality=100)
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0., val_split=0.2,
                          shuffle=True, preprocess=["make_lowercase", "clean_text"], drop_duplicates=True)
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
        print(incorrect_df)

    def test_lof_outlier_detection(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        print("module %s loaded" % module_url)

        dataset = Dataset(csv_path=f"{os.path.dirname(__file__)}/../data/saved/descriptions_25.csv", test_split=0.4,
                          val_split=0.2, shuffle=True, preprocess=["make_lowercase", "clean_text"], drop_duplicates=True)
        anvil_df = dataset.__data.loc[dataset.__data["label"] == 1]
        with tf.device('/CPU:0'):
            embeddings = model(anvil_df["description"].tolist())
            print(embeddings.numpy())
            clf = LocalOutlierFactor(n_neighbors=25)
            anvil_df["lof"] = clf.fit_predict(embeddings)
            outliers = anvil_df.loc[anvil_df["lof"] == -1].drop(["label"], axis=1).reset_index(drop=True)
            for outlier in outliers["description"].tolist():
                print(outlier)
            print(len(outliers))

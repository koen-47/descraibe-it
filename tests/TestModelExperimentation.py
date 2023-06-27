import os.path
import unittest

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.LSTM import LSTM


class TestModelExperimentation(unittest.TestCase):
    def test_dataset(self):
        glove = GloVeEmbedding(f"{os.path.dirname(__file__)}/../data/embeddings/glove.6B.100d.txt", dimensionality=100)
        dataset1 = Dataset(csv_path=f"{os.path.dirname(__file__)}/../data/saved/descriptions_25.csv", test_split=0.4,
                           val_split=0.2, shuffle=True, pipeline=["make_lowercase", "clean_text"], drop_duplicates=True)

        model1 = LSTM(dataset1, embedding=glove)
        hyperparameters = {
            "n_lstm_units": {"min": 16, "max": 512, "step": 32},
            "n_fc_layers": {"min": 1, "max": 5, "step": 1},
            "n_fc_units": {"min": 16, "max": 2048, "step": 64},
            "dropout_p": {"min": 0., "max": 0.9, "step": 0.1}
        }
        model1.start_tuning(n_trials=25, hyperparameters=hyperparameters)


# dataset2 = Dataset(csv_path=f"{os.path.dirname(__file__)}/../data/saved/descriptions_25.csv", test_split=0.4,
#                    val_split=0.2, shuffle=True,
#                    pipeline=["make_lowercase", "clean_text", "remove_stopwords"],
#                    drop_duplicates=True)
# dataset3 = Dataset(csv_path=f"{os.path.dirname(__file__)}/../data/saved/descriptions_25.csv", test_split=0.4,
#                    val_split=0.2, shuffle=True, pipeline=["make_lowercase", "clean_text", "remove_stopwords", "lemmatize"],
#                    drop_duplicates=True)

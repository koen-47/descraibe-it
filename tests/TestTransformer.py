import os
import unittest

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.Transformer import Transformer


class TestTransformer(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"
        self.embedding_filepath = f"{os.path.dirname(__file__)}/../data/embeddings"

    def test_transformer(self):
        pipeline = ["make_lowercase", "expand_contractions", "clean_text"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, preprocess=pipeline, drop_duplicates=True)
        glove = GloVeEmbedding(f"{self.embedding_filepath}/glove.6B.100d.txt", dimensionality=100)

        model = Transformer(dataset, glove, params={})
        model.fit()

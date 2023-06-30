import os
import unittest

from data.Dataset import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"

    def test_cv_split(self):
        pipeline = ["make_lowercase", "expand_contractions" "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)

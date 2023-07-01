import os
import unittest

from data.Dataset import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.data_filepath = f"{os.path.dirname(__file__)}/../data/saved"
        self.pipeline = ["make_lowercase", "expand_contractions" "clean_text", "remove_stopwords"]
        self.dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.3, val_split=0.2,
                               shuffle=True, pipeline=self.pipeline, drop_duplicates=True)

    def test_dataset_csv(self):
        print(self.dataset.train)
        print(self.dataset.val)
        print(self.dataset.test)

    def test_dataset_set_data(self):
        train_data = self.dataset.train[:1000]
        val_data = self.dataset.val[:1000]
        test_data = self.dataset.test[:1000]
        dataset = Dataset(train_data=train_data, val_data=val_data, test_data=test_data, pipeline=self.pipeline,
                          encode_labels=False, drop_duplicates=True)

        print(dataset.val["label"].value_counts())
        print(val_data["label"].value_counts())

    def test_dataset_definition_errors(self):
        train_data = self.dataset.train[:1000]
        val_data = self.dataset.val[:1000]
        test_data = self.dataset.test[:1000]
        dataset = Dataset(csv_path="", train_data=train_data, val_data=val_data, test_data=test_data,
                          pipeline=self.pipeline, encode_labels=False, drop_duplicates=True)

    def test_cv_split(self):
        pipeline = ["make_lowercase", "expand_contractions" "clean_text", "remove_stopwords"]
        dataset = Dataset(csv_path=f"{self.data_filepath}/descriptions_25.csv", test_split=0.4, val_split=0.2,
                          shuffle=True, pipeline=pipeline, drop_duplicates=True)
        cv = dataset.get_cv_split(n_splits=4)
        print(cv)
        for split in cv:
            print(len(split["train"]))
            print(len(split["test"]))

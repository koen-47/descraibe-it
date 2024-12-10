"""
File to handle all necessary functionality and data handling related to the training, test and validation data.
"""

import os
import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from .PreprocessingPipeline import PreprocessingPipeline


class Dataset:
    """
    Class to handle all necessary functionality and data handling related to the training, test and validation data.
    """

    def __init__(self, csv_path=None, train_data=None, val_data=None, test_data=None, test_split=0., preprocess=None,
                 val_split=0., shuffle=True, drop_duplicates=True, encode_labels=True):
        """
        Constructor for the Dataset class.

        :param csv_path: file path to CSV file containing the complete dataset (merged train, test and validation sets).
        Only usable if train_data, val_data and test_data are None.
        :param train_data: Pandas DataFrame with train set. Only usable if test_data is not None and csv_path
        is set to None.
        :param val_data: Pandas DataFrame with validation set. Only usable if val_split is 0 and csv_path
        is set to None.
        :param test_data: Pandas DataFrame with test set. Only usable if test_split is 0, train_data is not None and
        csv_path is set to None.
        :param test_split: Amount of data [0-1] to pass to the test set.
        :param preprocess: Preprocessing pipeline (list of strings)
        :param val_split: Amount of data [0-1] of the data to pass the validation from the train set.
        :param shuffle: Boolean to indicate if the data will be shuffled.
        :param drop_duplicates: Boolean to indicate if the duplicates will be dropped from the data.
        :param encode_labels: Boolean to indicate if the labels will be encoded using label encoding.
        """
        self.__test_split = test_split
        self.__val_split = val_split
        self.__shuffle = shuffle

        # Handling incorrect attribute settings
        if test_data is not None and test_split > 0.:
            raise ValueError("Test split cannot be set because test data has been defined.")
        if val_data is not None and val_split > 0.:
            raise ValueError("Validation split cannot be set because validation data has been defined.")
        if (train_data is not None and test_data is None) or (train_data is None and test_data is not None):
            raise ValueError("Both training and test data need to be defined.")
        if (train_data is not None or val_data is not None or test_data is not None) and csv_path is not None:
            raise ValueError("Unable to define dataset .csv filepath while data is defined.")

        # Set up preprocessing pipeline
        preprocess = preprocess if preprocess is not None else ["make_lowercase", "expand_contractions",
                                                                "remove_stopwords", "lemmatize", "clean_text"]

        # Read and preprocess all the data (if the data is not set already during instantiation)
        if train_data is None and val_data is None and test_data is None:
            self.__data = pd.read_csv(csv_path)
            self.__data = PreprocessingPipeline(preprocess, dataset=self.__data, shuffle=self.__shuffle).apply()
            self.__data["label"] = self.__encode_labels(self.__data["label"]) if encode_labels else self.__data["label"]

            # Perform validation split (necessary)
            if val_split > 0.:
                self.train, self.test, self.val = self.__train_test_val_split()
            else:
                self.train, self.test = self.__train_test_val_split()
                self.val = None
        else:
            # Preprocess the training, test and validation data if they are passed to the Dataset object during
            # instantiation
            print("Preprocessing train data...")
            self.train = PreprocessingPipeline(pipeline=preprocess, dataset=train_data).apply()
            print("\nPreprocessing test data...")
            self.test = PreprocessingPipeline(pipeline=preprocess, dataset=test_data).apply()
            if val_data is not None:
                print("\nPreprocessing validation data...")
                self.val = PreprocessingPipeline(pipeline=preprocess, dataset=val_data).apply()
            else:
                self.val = None

            # Encode the labels (i.e., words to describe) using label encoding
            if encode_labels:
                self.__encode_labels(self.get_full_dataset()["label"])
                self.train["label"] = self.label_encoder.transform(self.train["label"])
                self.test["label"] = self.label_encoder.transform(self.test["label"])
                if val_data is not None:
                    self.val["label"] = self.label_encoder.transform(self.val["label"])

        # Drop duplicates by keeping the first encountered duplicate (performed after preprocessing)
        if drop_duplicates:
            self.train = self.train.drop_duplicates(keep="first").reset_index(drop=True)
            self.test = self.test.drop_duplicates(keep="first").reset_index(drop=True)
            if self.__val_split > 0.:
                self.val = self.val.drop_duplicates(keep="first").reset_index(drop=True)

    def __train_test_val_split(self):
        """
        Splits the complete dataset (passed during instantiation) into train, test and (if necessary) validation sets.
        Uses the ratios defined during instantiation.

        :return: tuple consisting of the train, test and (if necessary) validation sets.
        """

        # Split training data
        train = self.__data[:int(len(self.__data) * (1 - self.__test_split))]

        # Split test data
        test = self.__data[len(train):]

        # Split validation (if specified by attribute)
        if self.__val_split > 0:
            temp = train
            train = train[:int(len(train) * (1 - self.__val_split))]
            val = temp[len(train):]
            return train, test, val
        return train, test

    def __encode_labels(self, labels):
        """
        Encodes the specified labels using label encoding. This will also save the computed label encodings to the file
        "models/saved/labels.npy"

        :param labels: specified labels to encode to strings.
        :return: encoded labels using label encoding.
        """
        self.label_encoder = LabelEncoder()
        encoded = self.label_encoder.fit_transform(labels)
        np.save(f"{os.path.dirname(__file__)}/../models/saved/labels.npy", self.label_encoder.classes_)
        return encoded

    def get_full_dataset(self):
        """
        Merges the train, test and (if necessary) validation sets and returns it as a Pandas DataFrame.

        :return: Pandas DataFrame containing the merged train, test and validation sets.
        """
        if self.val is not None:
            return pd.concat([self.train, self.val, self.test], ignore_index=True)
        return pd.concat([self.train, self.test], ignore_index=True)

    def get_all_words(self):
        """
        Class function to get all the words that are not whitespaces or empty spaces in the full data.
        :return: Returns all the words that are not whitespaces or empty spaces in the full data.
        """
        words = []
        for row in self.__data["description"]:
            for word in row.split(" "):
                if word != " " and word != "":
                    words.append(word)
        return words

    def decode_label(self, label):
        """
        Decodes an encoded label back to its original string using the LabelEncoder class
        :param label: Array of strings that contains the encoded labels as integers.
        :return: Returns an array of strings that contains the decoded labels as strings.
        """
        return self.label_encoder.inverse_transform(label)

    def __str__(self):
        """
        Class function used for printing.
        :return: String representation of the Pandas Dataframe containing the full data.
        """
        return str(self.__data)

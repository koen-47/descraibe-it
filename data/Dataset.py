import os
import re

from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from .PreprocessingPipeline import PreprocessingPipeline


class Dataset:
    """
    Class to handle all necessary functionality and data handling related to the training, test and validation data.
    """

    def __init__(self, csv_path=None, train_data=None, val_data=None, test_data=None, test_split=0., pipeline=None,
                 val_split=0., shuffle=False, drop_duplicates=False, encode_labels=True, random_state=42):
        """
        Constructor for the Dataset class.
        :param csv_path: Path to the CSV file that contains all the data.
        :param test_split: Fraction of the full data to split into the test data.
        :param val_split: Fraction of the training data to be split into the validation data.
        :param shuffle: Boolean to determine if the full data should be shuffled before splitting.
        :param remove_stopwords: Boolean to determine if stopwords (e.g., 'a', 'the', etc.) should be removed during
        preprocessing.
        """
        self.__test_split = test_split
        self.__val_split = val_split
        self.__shuffle = shuffle

        if test_data is not None and test_split > 0.:
            raise ValueError("Test split cannot be set because test data has been defined.")
        if val_data is not None and val_split > 0.:
            raise ValueError("Validation split cannot be set because validation data has been defined.")
        if train_data is not None and shuffle is True:
            raise ValueError("Unable to shuffle training data because training data is defined.")
        if (train_data is not None and test_data is None) or (train_data is None and test_data is not None):
            raise ValueError("Both training and test data need to be defined.")
        if (train_data is not None or val_data is not None or test_data is not None) and csv_path is not None:
            raise ValueError("Unable to define dataset .csv filepath while data is defined.")

        pipeline = pipeline if pipeline is not None else ["make_lowercase", "expand_contractions", "clean_text",
                                                          "remove_duplicates"]
        if train_data is None and val_data is None and test_data is None:
            self.__data = pd.read_csv(csv_path)
            self.__data = PreprocessingPipeline(self.__data, pipeline=pipeline).apply()
            self.__data["label"] = self.__encode_labels(self.__data["label"]) if encode_labels else self.__data["label"]
            if self.__shuffle:
                self.__data = self.__data.sample(frac=1).reset_index(drop=True)
            if val_split > 0.:
                self.train, self.val, self.test = self.__train_test_val_split()
            else:
                self.train, self.test = self.__train_test_val_split()
                self.val = None
        else:
            self.train = PreprocessingPipeline(train_data, pipeline=pipeline).apply()
            self.test = PreprocessingPipeline(test_data, pipeline=pipeline).apply()
            if val_data is not None:
                self.val = PreprocessingPipeline(val_data, pipeline=pipeline).apply()
            else:
                self.val = None
            if encode_labels:
                self.__encode_labels(self.get_full_dataset()["label"])
                self.train["label"] = self.label_encoder.transform(self.train["label"])
                self.test["label"] = self.label_encoder.transform(self.test["label"])
                if val_data is not None:
                    self.val["label"] = self.label_encoder.transform(self.val["label"])

        if drop_duplicates:
            self.train = self.train.drop_duplicates(keep="first").reset_index(drop=True)
            self.test = self.test.drop_duplicates(keep="first").reset_index(drop=True)
            if self.__val_split > 0.:
                self.val = self.val.drop_duplicates(keep="first").reset_index(drop=True)

    def __train_test_val_split(self):
        train = self.__data[:int(len(self.__data) * (1 - self.__test_split))]
        test = self.__data[len(train):]
        if self.__val_split > 0:
            temp = train
            train = train[:int(len(train) * (1 - self.__val_split))]
            val = temp[len(train):]
            return train, test, val
        return train, test

    def __encode_labels(self, labels):
        self.label_encoder = LabelEncoder()
        encoded = self.label_encoder.fit_transform(labels)
        np.save(f"{os.path.dirname(__file__)}/../models/saved/labels.npy", self.label_encoder.classes_)
        return encoded

    def get_cv_split(self, n_splits):
        data = self.get_full_dataset()
        kf = KFold(n_splits=n_splits)
        return [{"train": data.iloc[split[0]], "test": data.iloc[split[1]]} for split in list(kf.split(data))]

    def clean_text(self, text):
        """
        Class function that cleans all the data by removing punctuation, hyperlinks, double whitespaces, etc.
        :param text: Raw input text split into sentences.
        :param remove_stopwords: Boolean to determine if stopwords should be removed during cleaning.
        :return: Returns a Pandas Series that contains the cleaned text.
        """
        clean_text = []
        for sentence in text:
            sentence = sentence.lower()
            sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'\<a href', ' ', sentence)
            sentence = re.sub(r'&amp;', '', sentence)
            sentence = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)
            sentence = re.sub(r'<br />', ' ', sentence)
            sentence = re.sub(r'\'', ' ', sentence)
            sentence = re.sub(r'^(\d{1,2})(.|\)) ', '', sentence)
            sentence = re.sub(r'  ', ' ', sentence)

            # if self.remove_stopwords:
            #     sentence = sentence.split()
            #     stops = set(stopwords.words("english"))
            #     sentence = [w for w in sentence if not w in stops]
            #     sentence = " ".join(sentence)

            clean_text.append(sentence)
        return pd.Series(clean_text)

    def __count_vocab_size(self):
        """
        Private class function that counts the number of unique words (vocabulary) in the dataset.
        :return: Vocabulary size
        """
        unique_words = []
        for row in self.__data["description"]:
            for word in row.split(" "):
                if word not in unique_words and word != " " and word != "":
                    unique_words.append(word)
        return len(unique_words)

    def get_full_dataset(self):
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

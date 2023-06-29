import os
import re

from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .PreprocessingPipeline import PreprocessingPipeline


class Dataset:
    """
    Class to handle all necessary functionality and data handling related to the training, test and validation data.
    """

    def __init__(self, csv_path, test_split, pipeline=None, val_split=0., shuffle=False, drop_duplicates=False):
        """
        Constructor for the Dataset class.
        :param csv_path: Path to the CSV file that contains all the data.
        :param test_split: Fraction of the full data to split into the test data.
        :param val_split: Fraction of the training data to be split into the validation data.
        :param shuffle: Boolean to determine if the full data should be shuffled before splitting.
        :param remove_stopwords: Boolean to determine if stopwords (e.g., 'a', 'the', etc.) should be removed during
        preprocessing.
        """
        self.test_split = test_split
        self.val_split = val_split
        self.shuffle = shuffle
        # self.remove_stopwords = remove_stopwords

        self.data = pd.read_csv(csv_path)

        pipeline = pipeline if pipeline is not None else ["make_lowercase", "expand_contractions",
                                                          "clean_text", "remove_duplicates"]
        self.data = PreprocessingPipeline(self.data, pipeline=pipeline).apply()

        self.data["label"] = self.__encode_labels(self.data["label"])

        if val_split > 0.:
            self.train, self.val, self.test = self.__train_test_val_split()
        else:
            self.train, self.test = self.__train_test_val_split()

        if drop_duplicates:
            self.train = self.train.drop_duplicates(keep="first").reset_index(drop=True)
            self.test = self.test.drop_duplicates(keep="first").reset_index(drop=True)
            if self.val_split > 0.:
                self.val = self.val.drop_duplicates(keep="first").reset_index(drop=True)

    def __train_test_val_split(self):
        train = self.data[:int(len(self.data) * (1 - self.test_split))]
        test = self.data[len(train):]
        if self.val_split > 0:
            temp = train
            train = train[:int(len(train) * (1 - self.val_split))]
            val = temp[len(train):]
            return train, test, val
        return train, test

    def __encode_labels(self, labels):
        self.label_encoder = LabelEncoder()
        encoded = self.label_encoder.fit_transform(labels)
        np.save(f"{os.path.dirname(__file__)}/../models/saved/labels.npy", self.label_encoder.classes_)
        return encoded

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
        for row in self.data["description"]:
            for word in row.split(" "):
                if word not in unique_words and word != " " and word != "":
                    unique_words.append(word)
        return len(unique_words)

    def get_all_words(self):
        """
        Class function to get all the words that are not whitespaces or empty spaces in the full data.
        :return: Returns all the words that are not whitespaces or empty spaces in the full data.
        """
        words = []
        for row in self.data["description"]:
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
        return self.__label_encoder.inverse_transform(label)

    def __str__(self):
        """
        Class function used for printing.
        :return: String representation of the Pandas Dataframe containing the full data.
        """
        return str(self.data)

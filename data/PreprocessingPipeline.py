"""
File to handle all functionality necessary to preprocess the descriptions of each word.
"""

import json
import os.path
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
from tqdm import tqdm

# Download datasets from NLTK to help during preprocessing
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class PreprocessingPipeline:
    """
    Preprocessing pipeline that handles all functionality to preprocess the descriptions of each word.
    """
    def __init__(self, pipeline, dataset=None, feature="description", shuffle=True):
        """
        Constructor for the PreprocessingPipeline class.

        :param pipeline: list of strings denoting which preprocessing operations to perform and in which order.
        These operations are either: [make_lowercase, clean_text, remove_stopwords, expand_contractions, lemmatize].
        :param dataset: dataset to apply the preprocessing pipeline to (Dataset object)
        :param feature: target feature of the dataset object to apply the preprocessing pipeline to.
        :param shuffle: Boolean to indicate if the dataset will be shuffled during preprocessing.
        """
        self.dataset = dataset
        self.pipeline = pipeline
        self.feature = feature

        # Shuffle dataset (if specified)
        if shuffle and dataset is not None:
            self.dataset = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    def apply(self, x=None):
        """
        Applies the preprocessing pipeline to the target feature of the dataset, or to a specified value.
        :param x: specified value to apply the preprocessing pipeline to.
        :return: preprocessed dataset or specified value.
        """
        # Check to perform preprocessing on specified parameter on the entire dataset for a specific feature.
        if x is None:
            # Apply preprocessing to a feature in the dataset.
            for process_type in self.pipeline:
                self.dataset[self.feature] = self.__preprocess(self.dataset[self.feature], process_type)
            return self.dataset
        else:
            # Apply preprocessing to the specified value.
            for process_type in self.pipeline:
                x = self.__preprocess(x, process_type)
            return x

    def __preprocess(self, data, process_type):
        """
        Helper function that maps each string for a preprocessing step to its corresponding operation.
        :param data: data to apply the preprocessing pipeline to.
        :param process_type: type of operation to apply.
        :return: list of preprocessed data according to the specified operation to apply.
        """
        if process_type == "make_lowercase":
            return self.__make_lowercase(data)
        elif process_type == "clean_text":
            return self.__clean_text(data)
        elif process_type == "remove_stopwords":
            return self.__remove_stopwords(data)
        elif process_type == "expand_contractions":
            return self.__expand_contractions(data)
        elif process_type == "lemmatize":
            return self.__lemmatize(data)

    def __make_lowercase(self, data):
        """
        Converts the specified list of data to lowercase text.
        :param data: list of data to turn to lowercase text.
        :return: list of data that is converted to lowercase text.
        """
        lowercase = []

        # Iterate over all entries in the specified data.
        for sentence in tqdm(data, desc="Converting to lowercase"):
            # Convert the whole sentence to lowercase.
            sentence = sentence.lower()
            lowercase.append(sentence)

        return lowercase

    def __clean_text(self, data):
        """
        Cleans the text by removing punctuation, hyperlinks, etc.
        :param data: list of strings to clean.
        :return: list of cleaned strings.
        """
        clean_text = []

        # Iterate over all entries in the specified data.
        for sentence in tqdm(data, desc="Cleaning text"):
            # Remove URLs
            sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'\<a href', ' ', sentence)

            # Remove ampersands
            sentence = re.sub(r'&amp;', '', sentence)

            # Remove punctuation
            sentence = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)

            # Remove HTML link breaks
            sentence = re.sub(r'<br />', ' ', sentence)

            # Remove backslashes
            sentence = re.sub(r'\'', ' ', sentence)

            # Remove the start of the sentence if it starts with a number followed by ')'
            sentence = re.sub(r'^(\d{1,2})(.|\)) ', '', sentence)

            # Remove consecutive whitespaces
            sentence = re.sub(r'  ', ' ', sentence)

            clean_text.append(sentence.strip())

        return clean_text

    def __remove_stopwords(self, data):
        removed_stopwords = []

        # Load a dataset of stopwords
        stops = set(stopwords.words("english"))

        #  Iterate over all entries in the specified data.
        for sentence in tqdm(data, desc="Removing stopwords"):
            # Split sentence and remove all words that are a stopword
            sentence = sentence.split()
            sentence = [w for w in sentence if not w in stops]
            sentence = " ".join(sentence)
            removed_stopwords.append(sentence)

        return removed_stopwords

    def __expand_contractions(self, data):
        with open(f"{os.path.dirname(__file__)}/preprocessing/contractions_dict.json") as file:
            contractions = json.load(file)
        expanded_contractions = []
        for sentence in tqdm(data, desc="Expanding contractions"):
            sentence = sentence.split()
            sentence = [contractions.get(item, item) for item in sentence]
            sentence = " ".join(sentence)
            expanded_contractions.append(sentence)
        return pd.Series(expanded_contractions)

    def __lemmatize(self, data):
        # Load the WordNet lemmatizer
        lemmatizer = WordNetLemmatizer()
        lemmatized = []

        # Iterate over all entries in the specified data.
        for sentence in tqdm(data, desc="Lemmatizing"):
            # Tokenize the entry using NLTK
            sentence = nltk.word_tokenize(sentence)

            # Lemmatize the tokenized entry using POS tags
            sentence = [lemmatizer.lemmatize(token, self.__get_wordnet_pos(pos))
                        for token, pos in nltk.pos_tag(sentence)]

            sentence = " ".join(sentence)
            lemmatized.append(sentence)
        return lemmatized

    def __get_wordnet_pos(self, pos):
        """
        Helper function that maps an POS to its WordNet equivalent.
        :param pos: POS to convert to WordNet equivalent
        :return: WordNet POS equivalent of the specified POS.
        """
        # Adjectives
        if pos.startswith('J'):
            return wordnet.ADJ
        # Verbs
        elif pos.startswith('V'):
            return wordnet.VERB
        # Nouns
        elif pos.startswith('N'):
            return wordnet.NOUN
        # Adverbs
        elif pos.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

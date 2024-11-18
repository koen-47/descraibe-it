import json
import os.path
import re
from abc import ABC, abstractmethod
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class PreprocessingPipeline(ABC):
    def __init__(self, dataset, pipeline, feature="description", shuffle=True):
        self.dataset = dataset
        self.pipeline = pipeline
        self.feature = feature

        if shuffle:
            self.dataset = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    def apply(self):
        for preprocess in self.pipeline:
            self.__preprocess(preprocess)
        return self.dataset

    def __preprocess(self, preprocess):
        if preprocess == "make_lowercase":
            self.__make_lowercase()
        elif preprocess == "clean_text":
            self.__clean_text()
        elif preprocess == "remove_stopwords":
            self.__remove_stopwords()
        elif preprocess == "expand_contractions":
            self.__expand_contractions()
        elif preprocess == "lemmatize":
            self.__lemmatize()
        elif preprocess == "remove_duplicates":
            self.__remove_duplicates()

    def __make_lowercase(self):
        lowercase = []
        for sentence in tqdm(self.dataset[self.feature], desc="Converting to lowercase"):
            sentence = sentence.lower()
            lowercase.append(sentence)
        self.dataset[self.feature] = pd.Series(lowercase)

    def __clean_text(self):
        clean_text = []
        for sentence in tqdm(self.dataset[self.feature], desc="Cleaning text"):
            sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'\<a href', ' ', sentence)
            sentence = re.sub(r'&amp;', '', sentence)
            sentence = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)
            sentence = re.sub(r'<br />', ' ', sentence)
            sentence = re.sub(r'\'', ' ', sentence)
            sentence = re.sub(r'^(\d{1,2})(.|\)) ', '', sentence)
            sentence = re.sub(r'  ', ' ', sentence)
            clean_text.append(sentence)
        self.dataset[self.feature] = pd.Series(clean_text)

    def __remove_stopwords(self):
        removed_stopwords = []
        stops = set(stopwords.words("english"))
        for sentence in tqdm(self.dataset[self.feature], desc="Removing stopwords"):
            sentence = sentence.split()
            sentence = [w for w in sentence if not w in stops]
            sentence = " ".join(sentence)
            removed_stopwords.append(sentence)
        self.dataset[self.feature] = pd.Series(removed_stopwords)

    def __expand_contractions(self):
        with open(f"{os.path.dirname(__file__)}/preprocessing/contractions_dict.json") as file:
            contractions = json.load(file)
        expanded_contractions = []
        for sentence in tqdm(self.dataset[self.feature], desc="Expanding contractions"):
            sentence = sentence.split()
            sentence = [contractions.get(item, item) for item in sentence]
            sentence = " ".join(sentence)
            expanded_contractions.append(sentence)
        self.dataset[self.feature] = pd.Series(expanded_contractions)

    def __lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        lemmatized = []
        for sentence in tqdm(self.dataset[self.feature], desc="Lemmatizing"):
            sentence = nltk.word_tokenize(sentence)
            sentence = [lemmatizer.lemmatize(token, self.__get_wordnet_pos(pos))
                        for token, pos in nltk.pos_tag(sentence)]
            sentence = " ".join(sentence)
            lemmatized.append(sentence)
        self.dataset[self.feature] = pd.Series(lemmatized)

    def __get_wordnet_pos(self, pos):
        if pos.startswith('J'):
            return wordnet.ADJ
        elif pos.startswith('V'):
            return wordnet.VERB
        elif pos.startswith('N'):
            return wordnet.NOUN
        elif pos.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def __remove_duplicates(self):
        self.dataset.drop_duplicates(inplace=True)

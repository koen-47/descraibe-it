import json
import os.path
import re
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import pandas as pd
import nltk
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


class PreprocessingPipeline:
    def __init__(self, pipeline, dataset=None, feature="description", shuffle=True):
        self.dataset = dataset
        self.pipeline = pipeline
        self.feature = feature

        if shuffle and dataset is not None:
            self.dataset = self.dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    def apply(self, x=None):
        if x is None:
            for process_type in self.pipeline:
                self.dataset[self.feature] = self.__preprocess(self.dataset[self.feature], process_type)
            return self.dataset
        else:
            for process_type in self.pipeline:
                x = self.__preprocess(x, process_type)
            return x

    def __preprocess(self, data, process_type):
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
        lowercase = []
        for sentence in tqdm(data, desc="Converting to lowercase"):
            sentence = sentence.lower()
            lowercase.append(sentence)
        return lowercase

    def __clean_text(self, data):
        clean_text = []
        for sentence in tqdm(data, desc="Cleaning text"):
            sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'\<a href', ' ', sentence)
            sentence = re.sub(r'&amp;', '', sentence)
            sentence = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)
            sentence = re.sub(r'<br />', ' ', sentence)
            sentence = re.sub(r'\'', ' ', sentence)
            sentence = re.sub(r'^(\d{1,2})(.|\)) ', '', sentence)
            sentence = re.sub(r'  ', ' ', sentence)
            clean_text.append(sentence.strip())
        return clean_text

    def __remove_stopwords(self, data):
        removed_stopwords = []
        stops = set(stopwords.words("english"))
        for sentence in tqdm(data, desc="Removing stopwords"):
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
        lemmatizer = WordNetLemmatizer()
        lemmatized = []
        for sentence in tqdm(data, desc="Lemmatizing"):
            sentence = nltk.word_tokenize(sentence)
            sentence = [lemmatizer.lemmatize(token, self.__get_wordnet_pos(pos))
                        for token, pos in nltk.pos_tag(sentence)]
            sentence = " ".join(sentence)
            lemmatized.append(sentence)
        return lemmatized

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

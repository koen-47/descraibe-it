import json
import os.path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from data.Dataset import Dataset
from data.GloVeEmbedding import GloVeEmbedding
from models.kNN import kNN
from models.SVM import SVM


class TestEmbeddingModels(unittest.TestCase):
    def setUp(self) -> None:
        train_data = pd.read_csv("../data/splits/train.csv")
        test_data = pd.read_csv("../data/splits/test.csv")
        val_data = pd.read_csv("../data/splits/val.csv")
        pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
        self.dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline)

    def test_svm_glove_embeddings(self):
        glove = GloVeEmbedding(f"../data/embeddings/glove.6B.100d.txt", dimensionality=100).embedding_index
        x_train = self.dataset.train["description"]
        x_train = np.array([self.get_sentence_embedding(text, glove) for text in x_train])
        y_train = self.dataset.train["label"]

        x_val = self.dataset.val["description"]
        x_val = np.array([self.get_sentence_embedding(text, glove) for text in x_val])
        y_val = self.dataset.val["label"]

        svm = SVC( random_state=42)
        svm.fit(x_train, y_train)

        y_pred = svm.predict(x_val)
        print("Accuracy:", accuracy_score(y_val, y_pred))

    def get_sentence_embedding(self, sentence, embeddings, embedding_dim=100):
        words = sentence.split()
        word_vectors = [embeddings[word] for word in words if word in embeddings]
        if not word_vectors:
            return np.zeros(embedding_dim)
        return np.mean(word_vectors, axis=0)

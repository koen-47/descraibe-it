import os
import random

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm


class GloVeEmbedding:
    """
    Class that handles all functionality related to GloVe embeddings.
    """

    def __init__(self, file_path):
        """
        Constructor for the GloVe embedding class.
        :param file_path: File path to where the GloVe embeddings are stored.
        """
        self.embedding_index = self.__get_embedding_index(file_path)
        self.dimensionality = int(os.path.split(file_path)[1].split(".")[2][:-1])

    def __get_embedding_index(self, file_path):
        """
        Parses the file containing the GloVe embeddings and returns its word index.
        :param file_path: File path to where the GloVe embeddings are stored.
        :return: Dictionary containing the word index of the GloVe embeddings file.
        """
        embeddings_index = dict()
        file = open(file_path, encoding="utf8")
        num_lines = sum(1 for _ in open(file_path, encoding="utf-8"))
        for line in tqdm(file, total=num_lines, desc="Loading GloVe embedding"):
            values = line.strip().split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        file.close()
        return embeddings_index

    def compute_embedding_matrix(self, tokenizer, size_of_vocabulary):
        """
        Computes the embedding matrix to be used in the embedding layer of a neural model.
        :param tokenizer: Keras Tokenizer that has been fit to the training data.
        :param size_of_vocabulary: Vocabulary size (integer)
        :return: Returns the embedding matrix.
        """
        embedding_matrix = np.zeros((size_of_vocabulary, self.dimensionality))

        for word, i in tokenizer.word_index.items():
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def visualize_words(self, words, special_words=None):
        tsne = TSNE(n_components=2, random_state=0, perplexity=len(words) - 1)
        embedding_vectors = np.array([self.embedding_index[word] for word in words])
        embedding_vectors_2d = tsne.fit_transform(embedding_vectors)

        plt.figure(figsize=(8, 8))
        for i, word in enumerate(words):
            x, y = embedding_vectors_2d[i, :]
            color = "blue" if special_words is not None else None
            if special_words is not None and word in special_words:
                plt.scatter(x, y, color="red")
            else:
                plt.scatter(x, y, color=color)
            plt.annotate(word, (x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")

        plt.show()

    def calculate_k_words_max_min_distance(self, words, k, n=100):
        embedding_word_vectors = {word: self.embedding_index[word] for word in words}

        min_distances = []
        for _ in tqdm(range(n)):
            k_vectors = dict(random.sample(embedding_word_vectors.items(), k))
            distances = []
            for v1 in k_vectors.values():
                for v2 in k_vectors.values():
                    if not np.array_equal(v1, v2):
                        distance = euclidean(v1, v2)
                        distances.append(distance)
            min_distance = min(distances)
            min_distances.append((str(list(k_vectors.keys())), min_distance))

        min_distances = sorted(min_distances, key=lambda x: x[1], reverse=True)
        return min_distances

    def calculate_min_distance_between_words(self, words):
        embedding_word_vectors = {word: self.embedding_index[word] for word in words}

        distances = []
        for w1, v1 in embedding_word_vectors.items():
            for w2, v2 in embedding_word_vectors.items():
                if not np.array_equal(v1, v2):
                    distance = euclidean(v1, v2)
                    distances.append((w1, w2, distance))
        return min(distances, key=lambda item: item[2])

    def __getitem__(self, item):
        return self.embedding_index[item]

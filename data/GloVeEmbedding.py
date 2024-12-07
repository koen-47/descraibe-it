import os
import random

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import seaborn as sns


class GloVeEmbedding:
    """
    Class that handles all functionality related to GloVe embeddings.
    """

    def __init__(self, file_path, dimensionality):
        """
        Constructor for the GloVe embedding class.
        :param file_path: File path to where the GloVe embeddings are stored.
        """
        if file_path is not None:
            self.embedding_index = self.__get_embedding_index(file_path)
        self.dimensionality = dimensionality

    def __get_embedding_index(self, file_path):
        """
        Parses the file containing the GloVe embeddings and returns its word index.
        :param file_path: File path to where the GloVe embeddings are stored.
        :return: Dictionary containing the word index of the GloVe embeddings file.
        """
        embeddings_index = dict()
        with open(file_path, encoding="utf-8") as file:
            num_lines = sum(1 for _ in file)

        file = open(file_path, encoding="utf8")
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

    def visualize_words(self, words, special_words=None, dark_mode=True):
        tsne = TSNE(n_components=2, random_state=0, perplexity=len(words) - 1)
        embedding_vectors = np.array([self.embedding_index[word] for word in words])
        embedding_vectors_2d = tsne.fit_transform(embedding_vectors)

        plot_data = pd.DataFrame({
            "x": embedding_vectors_2d[:, 0],
            "y": embedding_vectors_2d[:, 1],
            "word": words,
            'color': ['#ff4747' if special_words is not None and word in special_words else '#459abd' for word in words],
            "alpha": [0.75 if special_words is not None and word in special_words else 0.25 for word in words]
        })

        fig = plt.figure(figsize=(12, 8))
        sns.set(font_scale=1.1)
        ax = sns.scatterplot(
            data=plot_data,
            x='x',
            y='y',
            hue='color',
            palette={c: c for c in plot_data['color'].unique()},
            legend=False,
            alpha=plot_data["alpha"]
        )

        color = "#0d1117" if not dark_mode else "#F0F6FC"
        if special_words is not None:
            for _, row in plot_data.iterrows():
                if row['word'] in special_words:
                    plt.annotate(row['word'], (row['x'], row['y']), xytext=(0, 12),
                                 textcoords="offset points", ha="center", va="center", color=color)

        ax.spines["bottom"].set_color(color)
        ax.spines["left"].set_color(color)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="x", colors=color)
        ax.tick_params(axis="y", colors=color)
        ax.grid(False)

        plt.xlabel("Embedding #1")
        plt.ylabel("Embedding #2")
        plt.savefig(f"../data/resources/word_selection_plot_{'dark' if dark_mode else 'light'}.png", transparent=True)

    # def calculate_k_words_max_min_distance(self, words, k, n=100):
    #     embedding_word_vectors = {word: self.embedding_index[word] for word in words}
    #
    #     min_distances = []
    #     for _ in tqdm(range(n)):
    #         k_vectors = dict(random.sample(embedding_word_vectors.items(), k))
    #         distances = []
    #         for v1 in k_vectors.values():
    #             for v2 in k_vectors.values():
    #                 if not np.array_equal(v1, v2):
    #                     distance = euclidean(v1, v2)
    #                     distances.append(distance)
    #         min_distance = min(distances)
    #         min_distances.append((str(list(k_vectors.keys())), min_distance))
    #
    #     min_distances = sorted(min_distances, key=lambda x: x[1], reverse=True)
    #     return min_distances

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

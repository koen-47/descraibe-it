import itertools
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from scipy.spatial.distance import euclidean

from data.GloVeEmbedding import GloVeEmbedding
from util import load_categories, find_maximal_subset


def print_distance_cosine_sim_arrays(avg_distances, med_distances, avg_cosine_sims, med_cosine_sims, n=5):
    print(f"Top {n} Average Euclidean Distances")
    print_distance_cosine_sim_array(avg_distances, n=n)
    print(f"\nTop {n} Median Euclidean Distances")
    print_distance_cosine_sim_array(med_distances, n=n)
    print(f"Top {n} Average Cosine Similarities")
    print_distance_cosine_sim_array(avg_cosine_sims, n=n)
    print(f"\nTop {n} Median Cosine Similarities")
    print_distance_cosine_sim_array(med_cosine_sims, n=n)


def print_distance_cosine_sim_array(array, n=5):
    for i in range(len(array[:n])):
        print(f"Number {i + 1}")
        print(f"   {array[i][0]}")
        print(f"   {array[i][1]}")


class TestGloVeEmbedding6B100d(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = GloVeEmbedding("../data/embeddings/glove.6B.100d.txt")

    def test_visualize_words_20(self):
        words = load_categories("../data/saved/categories_20.txt")
        print(words)
        self.embeddings.visualize_words(words)

    def test_visualize_words_100(self):
        words = load_categories("../data/saved/categories_100.txt")
        self.embeddings.visualize_words(words)

    def test_visualize_words_289_special_words_1(self):
        words = load_categories("../data/saved/categories_289.txt")
        special_words = ['mouth', 'mushroom', 'saxophone', 'hurricane', 'broccoli', 'guitar', 'triangle', 'bird',
                         'lightning', 'hexagon', 'clarinet', 'rhinoceros', 'stove', 'sun', 'spreadsheet', 'basketball',
                         'canoe', 'drums', 'computer', 'peanut', 'toothpaste', 'mug', 'church', 'hedgehog', 'sweater']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_visualize_words_289_special_words_2(self):
        words = load_categories("../data/saved/categories_289.txt")
        special_words = ['brain', 'oven', 'mushroom', 'dumbbell', 'diamond', 'spreadsheet', 'elephant', 'toe', 'sheep',
                         'keyboard', 'dresser', 'toothpaste', 'snorkel', 'dishwasher', 'pants', 'trombone', 'mountain',
                         'pliers', 'streetlight', 'crab', 'clarinet', 'sun', 'van', 'square', 'telephone']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_visualize_words_100_special_words_1(self):
        words = load_categories("../data/saved/categories_100.txt")
        special_words = ['apple', 'monkey', 'angel', 'toe', 'stairs', 'television', 'fence', 'door', 'beach', 'frog',
                         'crown', 'mouse', 'flower', 'sweater', 'foot', 'ear', 'diamond', 'horse', 'parrot', 'star',
                         'umbrella', 'whale', 'dragon', 'hat', 'nose']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_visualize_words_100_special_words_2(self):
        words = load_categories("../data/saved/categories_100.txt")
        special_words = ['bucket', 'rollerskates', 'house', 'baseball', 'flower', 'toilet', 'lobster', 'oven', 'lion',
                         'car', 'mountain', 'banana', 'dragon', 'shorts', 'door', 'ear', 'map', 'zebra', 'table',
                         'angel', 'truck', 'hat', 'diamond', 'apple', 'book']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_visualize_words_100_special_words_3(self):
        words = load_categories("../data/saved/categories_100.txt")
        special_words = ['squiggle', 'lion', 'dragon', 'teapot', 'bicycle', 'whale', 'nail', 'basketball', 'waterslide',
                         'oven', 'bulldozer', 'kangaroo', 'frog', 'cookie', 'submarine', 'jail', 'trombone', 'banana',
                         'crown', 'peas', 'computer', 'calculator', 'hexagon', 'mountain', 'book']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_visualize_words_100_special_words_4(self):
        words = load_categories("../data/saved/categories_100.txt")
        special_words = ['peas', 'butterfly', 'oven', 'stairs', 'postcard', 'whale', 'yacht', 'violin', 'crown',
                         'horse', 'nail', 'teapot', 'bulldozer', 'sword', 'triangle', 'television', 'mountain',
                         'basketball', 'matches', 'tent', 'peanut', 'pliers', 'submarine', 'toe', 'beach']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_calculate_k_most_distant_words_1(self):
        words = load_categories("../data/saved/categories_289.txt")
        min_distances = self.embeddings.calculate_k_words_max_min_distance(
            words, k=25, n=100000)
        print_distance_cosine_sim_array(min_distances)

    def test_greedy_calculate_k_most_distant_words_1(self):
        words = load_categories("../data/saved/categories_289.txt")
        vectors = self.embeddings.greedy_calculate_k_words_max_min_distance(words, k=5)
        self.embeddings.visualize_words(words, special_words=vectors)

    def test_calculate_min_distance_between_words(self):
        words = ["snail", "van"]
        min_distance = self.embeddings.calculate_min_distance_between_words(words)
        print(min_distance)

    def test_calculate_min_distance_between_words_manual_selection(self):
        words = load_categories("../data/saved/categories_289.txt")
        manual_words = ["snail", "van", "oven", "hurricane", "hat", "passport", "violin", "broccoli", "fish",
                        "hospital", "squirrel", "angel", "shovel", "toothbrush"]
        min_distance = self.embeddings.calculate_min_distance_between_words(manual_words)
        print(f"Min. distance: {min_distance}")
        print(f"Number of words: {len(manual_words)}")
        self.embeddings.visualize_words(words, special_words=manual_words)

    def test_temp(self):
        words = load_categories("../data/saved/categories_289.txt")
        embedding_word_vectors = {word: self.embeddings.embedding_index[word] for word in words}
        subset = find_maximal_subset(embedding_word_vectors)
        min_distance = self.embeddings.calculate_min_distance_between_words(subset)
        print(min_distance)
        self.embeddings.visualize_words(words, special_words=subset)


class TestGloVeEmbedding840B300d(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = GloVeEmbedding("../data/embeddings/glove.840B.300d.txt", dimensionality=300)
        # self.embeddings = GloVeEmbedding("../data/embeddings/glove.6B.100d.txt", dimensionality=100)

    def test_visualize_words_20(self):
        words = load_categories("../data/saved/categories_20.txt")
        self.embeddings.visualize_words(words)

    def test_visualize_words_100(self):
        words = load_categories("../data/saved/categories_100.txt")
        self.embeddings.visualize_words(words)

    def test_visualize_words_289(self):
        words = load_categories("../data/saved/categories_289.txt")
        self.embeddings.visualize_words(words)

    def test_visualize_words_289_special_words_1(self):
        words = load_categories("../data/saved/categories_289.txt")
        special_words = ['brain', 'oven', 'mushroom', 'dumbbell', 'diamond', 'spreadsheet', 'elephant', 'toe', 'sheep',
                         'keyboard', 'dresser', 'toothpaste', 'snorkel', 'dishwasher', 'pants', 'trombone', 'mountain',
                         'pliers', 'streetlight', 'crab', 'clarinet', 'sun', 'van', 'square', 'telephone']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_visualize_words_289_special_words_2(self):
        words = load_categories("../data/saved/categories_289.txt")
        special_words = ['vase', 'door', 'bus', 'sailboat', 'nose', 'toothbrush', 'book', 'map', 'radio', 'syringe',
                         'submarine', 'wristwatch', 'hospital', 'snowman', 'rhinoceros', 'grass', 'crown',
                         'streetlight', 'telephone', 'guitar', 'helmet', 'blueberry', 'ant', 'helicopter', 'megaphone']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_calculate_k_most_distant_words_1(self):
        words = load_categories("../data/saved/categories_289.txt")
        min_distances = self.embeddings.calculate_k_words_max_min_distance(
            words, k=25, n=100000)
        print_distance_cosine_sim_array(min_distances)

    def test_temp(self):
        words = load_categories("../data/saved/categories_289.txt")
        embedding_word_vectors = {word: self.embeddings.embedding_index[word] for word in words}
        subset = find_maximal_subset(embedding_word_vectors, k=25)
        min_distance = self.embeddings.calculate_min_distance_between_words(subset)
        print(subset)
        print(min_distance)
        chunks = np.array_split(subset, len(subset) // 25)
        for chunk in chunks:
            print(chunk)

        self.embeddings.visualize_words(words, special_words=subset)


    def visualize_words(self, words, special_words=None, dark_mode=True):
        tsne = TSNE(n_components=2, random_state=0, perplexity=len(words) - 1)
        embedding_vectors = np.array([self.embeddings.embedding_index[word] for word in words])
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

    def calculate_min_distance_between_words(self, words):
        embedding_word_vectors = {word: self.embeddings.embedding_index[word] for word in words}

        distances = []
        for w1, v1 in embedding_word_vectors.items():
            for w2, v2 in embedding_word_vectors.items():
                if not np.array_equal(v1, v2):
                    distance = euclidean(v1, v2)
                    distances.append((w1, w2, distance))
        return min(distances, key=lambda item: item[2])


import itertools
import unittest

import numpy as np

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

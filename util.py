"""
File containing utility functions.
"""

import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd

import warnings

warnings.simplefilter("default")

# Suppress DeprecationError regarding randrange (I'm not sure where this warning comes from and how to resolve it)
warnings.filterwarnings(
    "ignore",
    message=r"^non-integer arguments to randrange\(\).*deprecated since Python 3\.10.*$",
    category=DeprecationWarning
)


# Supress DeprecationWarning regarding using numpy using booleans (harmless warning)
warnings.filterwarnings(
    "ignore",
    message=r"`np.bool` is a deprecated alias",
    category=DeprecationWarning
)


def load_categories(file_path):
    """
    Loads the categories (i.e., words that need to be described) based on the specified file path.
    :param file_path: specified file path containing the words that need to be described.
    :return: list of words that need to be described.
    """
    categories = []
    with open(file_path) as file:
        for category in file:
            categories.append(category.strip())
    return categories


def find_maximal_subset(vector_dict, k=25):
    """
    Greedy approximation of the optimal solution for the max-min diversity problem.
    :param vector_dict: dictionary mapping each word to its corresponding pretrained GloVe embedding.
    :param k: number of words to select (i.e., size of W').
    :return: list of words that have their minimum diversity (approximately) maximized according to their embeddings.
    """

    # Get words and embeddings as separate lists.
    words = list(vector_dict.keys())
    vectors = list(vector_dict.values())

    selected_subset = [words[0]]

    # Compute distance matrix (Euclidean)
    distances = cdist([vectors[0]], vectors).min(axis=0)

    # Iterate k times
    for _ in range(1, k):
        # Compute the index of the most distant word from the previously selected word.
        max_distance_idx = np.argmax(distances)

        # Add it to the subset of words that will be selected.
        selected_subset.append(words[max_distance_idx])

        # Recompute distance matrix based on the newly selected word.
        new_distances = cdist([vectors[max_distance_idx]], vectors).min(axis=0)

        # Get the element-wise minimum between the old and newly computed distances.
        distances = np.minimum(distances, new_distances)

    return selected_subset

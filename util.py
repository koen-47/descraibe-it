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
    categories = []
    with open(file_path) as file:
        for category in file:
            categories.append(category.strip())
    return categories


def find_maximal_subset(vector_dict, k=25):
    words = list(vector_dict.keys())
    vectors = list(vector_dict.values())
    selected_subset = [words[0]]
    distances = cdist([vectors[0]], vectors).min(axis=0)

    for _ in range(1, k):
        max_distance_idx = np.argmax(distances)
        selected_subset.append(words[max_distance_idx])
        new_distances = cdist([vectors[max_distance_idx]], vectors).min(axis=0)
        distances = np.minimum(distances, new_distances)

    return selected_subset


def concat_csv_files_to_df(csv_files, save_csv_path=None):
    df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True)
    if save_csv_path is not None:
        df.to_csv(save_csv_path, index=False)
    return df

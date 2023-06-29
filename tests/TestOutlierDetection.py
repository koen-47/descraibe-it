import os
import unittest

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor

from data.Dataset import Dataset

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)


class TestOutlierDetection(unittest.TestCase):
    def test_use_outlier_detection(self):
        dataset = Dataset(csv_path=f"{os.path.dirname(__file__)}/../data/saved/descriptions_25.csv", test_split=0.4,
                          val_split=0.2, shuffle=True, pipeline=["make_lowercase", "clean_text"], drop_duplicates=True)
        anvil_df = dataset.data.loc[dataset.data["label"] == 1]
        with tf.device('/CPU:0'):
            embeddings = model(anvil_df["description"].tolist())
            print(embeddings.numpy())
            clf = LocalOutlierFactor(n_neighbors=25)
            anvil_df["lof"] = clf.fit_predict(embeddings)
            outliers = anvil_df.loc[anvil_df["lof"] == -1].drop(["label"], axis=1).reset_index(drop=True)
            for outlier in outliers["description"].tolist():
                print(outlier)
            print(len(outliers))

            # anvil_df["embedding"] = embeddings.numpy()
            # print(anvil_df)

            # tsne = TSNE(n_components=2, random_state=0, perplexity=len(embeddings) - 1)
            # embedding_vectors_2d = tsne.fit_transform(embeddings)
            #
            # print(embedding_vectors_2d)
            # plt.figure(figsize=(8, 8))
            # x = [emb[0] for emb in embedding_vectors_2d]
            # y = [emb[1] for emb in embedding_vectors_2d]
            # plt.scatter(x, y, color="red", s=2)
            # plt.show()

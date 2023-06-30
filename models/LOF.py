import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor


class LOF:
    def __init__(self, dataset, load_model_path=None):
        self.__dataset = dataset

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load("./models/saved/labels.npy", allow_pickle=True)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.__dataset.__data["description"])
        labels = self.__dataset.__data["label"]
        unique_labels = np.unique(labels)

        for label in unique_labels[:1]:
            class_indices = [index for index, lbl in enumerate(labels) if lbl == label]
            class_X = X[class_indices]

            lof = LocalOutlierFactor(n_neighbors=2)
            class_outlier_scores = lof.fit_predict(class_X)
            print(f"{label_encoder.inverse_transform([label])[0]}: {class_outlier_scores}")
            print([vectorizer.inverse_transform(pred) for pred in class_X])


import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


class IF:
    def __init__(self, dataset, load_model_path=None):
        self.__dataset = dataset
        self.__vectorizer = TfidfVectorizer()

        print("Fitting vectorizer...")
        X = self.__vectorizer.fit_transform(self.__dataset.__data["description"])
        target_label = 1
        instances_for_label = X[np.array(self.__dataset.__data["label"]) == target_label]
        print(instances_for_label)

        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load("./models/saved/labels.npy", allow_pickle=True)
        classes = np.unique(self.__dataset.__data["label"])
        for label in classes[:1]:
            class_instances = X[self.__dataset.__data["label"] == label]
            original_texts = self.__vectorizer.inverse_transform(class_instances)
            print(dict(zip(original_texts, label)))
            
            # clf = IsolationForest(contamination=0.05)
            # clf.fit(class_instances)
            #
            # print(label_encoder.inverse_transform([label]))
            # outlier_scores = clf.decision_function(class_instances)
            # print([(x, label, score) for x, score in zip(self.__dataset.data["description"], outlier_scores)])

            # threshold = outlier_scores.mean() + 2 * outlier_scores.std()
            # outliers = X[outlier_scores > threshold]
            # print(outliers)

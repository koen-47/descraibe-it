import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class SVM:
    def __init__(self, dataset, load_model_path=None):
        self.__dataset = dataset
        self.__vectorizer = TfidfVectorizer(max_features=1000)

        print("Fitting vectorizer...")
        self.__bow = self.__vectorizer.fit_transform(self.__dataset.data["description"])
        print("Finished vectorizing...")

        x_train, x_test, y_train, y_test = train_test_split(self.__bow, np.asarray(self.__dataset.data["label"]),
                                                            test_size=0.33)
        model = SVC()

        start_time = time.time()
        model.fit(x_train, y_train)
        print(f"Completed fitting: {time.time() - start_time}")

        print(model.score(x_test, y_test))
        print(f"Completed scoring: {time.time() - start_time}")

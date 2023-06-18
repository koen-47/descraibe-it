import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, anneal


class kNN:
    def __init__(self, dataset, load_model_path=None):
        self.__dataset = dataset
        self.__vectorizer = TfidfVectorizer(max_features=1000)

        print("Fitting vectorizer...")
        self.__bow = self.__vectorizer.fit_transform(self.__dataset.data["description"])
        print("Finished vectorizing...")

        x_train, x_test, y_train, y_test = train_test_split(self.__bow, np.asarray(self.__dataset.data["label"]),
                                                            test_size=0.33)
        model = KNeighborsClassifier()

        start_time = time.time()
        model.fit(x_train, y_train)
        print(f"Completed fitting: {time.time() - start_time}")

        print(model.score(x_test, y_test))
        print(f"Completed scoring: {time.time() - start_time}")

        # y_pred = model.predict(x_test)
        # print(accuracy_score(y_test, y_pred))
        # print(time.time() - start_time)

        # print("Starting Bayesian optimization...")
        # def objective(params):
        #     model = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]),
        #                                  weights=params["weights"],
        #                                  metric=params["metric"])
        #     error = cross_val_score(model, x_train, y_train, cv=10, scoring="accuracy")
        #     return {"loss": -error, "status": STATUS_OK}
        #
        # weights = ["uniform", "distance"]
        # metrics = ['euclidean', 'manhattan']
        # params = {
        #     "n_neighbors": hp.quniform("n_neighbors", 1, 100, 1),
        #     "weights": hp.choice("weights", weights),
        #     "metric": hp.choice("metric", metrics),
        # }
        #
        # trials = Trials()
        # best_params = fmin(fn=objective,
        #                    space=params,
        #                    algo=tpe.suggest,
        #                    max_evals=10,
        #                    trials=trials)
        # print(best_params)
        #
        # model = KNeighborsClassifier(n_neighbors=3)
        # model.fit(x_train, y_train)
        #
        # y_pred = model.predict(x_test)
        # knn = KNeighborsClassifier(n_neighbors=params["n_neighbors"],
        #                            weights=params["weights"],
        #                            metric=params["metric"])
        # knn.fit(x_train, y_train)
        # y_pred = knn.predict(x_test)
        # accuracy = accuracy_score(y_test, y_pred)
        #
        # print(accuracy)

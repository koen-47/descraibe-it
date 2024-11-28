import os
import argparse
import json

import pandas as pd

from data.GloVeEmbedding import GloVeEmbedding
from data.PromptManager import PromptManager
from data.Dataset import Dataset
from models.LSTM import LSTM
from models.kNN import kNN
from models.SVM import SVM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    model_type = args.model
    is_evaluating = args.evaluate
    is_tuning = args.tune

    train_data = pd.read_csv("./data/splits/train.csv")
    test_data = pd.read_csv("./data/splits/test.csv")
    val_data = pd.read_csv("./data/splits/val.csv")
    pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
    dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline)

    if model_type == "lstm":
        if is_evaluating:
            params = {
                "lstm_layers": [{"bidirectional": True, "units": 480}],
                "fc_layers": [{"units": 656, "dropout_p": 0.7}],
                "early_stopping": {"patience": 10, "verbose": 1},
                "scheduler": {"initial_learning_rate": 0.001, "decay_steps": 15},
                "optimizer": {"name": "adam", "beta_1": 0.9, "beta_2": 0.999},
                "misc": {"epochs": 500, "batch_size": 256, "save_filepath": "./models/saved/lstm.h5"}
            }

            glove = GloVeEmbedding(f"./data/embeddings/glove.840B.300d.txt", dimensionality=300)
            model = LSTM(dataset, embedding=glove, params=params)
            model.fit()
            model.evaluate(verbose=True)

    elif model_type == "knn":
        with open("./results/knn/knn_results.json") as file:
            knn_results = json.load(file)
        if is_evaluating:
            best_params = knn_results["tuning"]["best_params"]
            model = kNN(dataset, best_params)
            model.fit(use_val=True)
            model.evaluate(verbose=True)
            model.plot_confusion_matrix(show=True)
        elif is_tuning:
            param_space = knn_results["tuning"]["param_space"]
            model = kNN(dataset, {})
            best_params = model.tune(param_space, n_jobs=None, verbose=True)
            model = kNN(dataset, best_params)
            model.fit(use_val=True)
            model.evaluate(verbose=True)
            model.plot_confusion_matrix(show=True)
    elif model_type == "svm":
        with open("./results/svm/svm_results.json") as file:
            svm_results = json.load(file)
        if is_evaluating:
            best_params = svm_results["tuning"]["best_params"]
            model = SVM(dataset, best_params)
            model.fit(use_val=True)
            model.evaluate(verbose=True)
            model.plot_confusion_matrix(show=True)
        elif is_tuning:
            param_space = svm_results["tuning"]["param_space"]
            model = SVM(dataset, {})
            best_params = model.tune(param_space, n_jobs=None, verbose=True)
            model = SVM(dataset, best_params)
            model.fit(use_val=True)
            model.evaluate(verbose=True)
            model.plot_confusion_matrix(show=True)


# dataset = Dataset(csv_path="./data/saved/descriptions_25.csv", test_split=0.4, val_split=0.2, shuffle=True)
# glove = GloVeEmbedding("./data/embeddings/glove.6B.100d.txt", dimensionality=100)
# glove = GloVeEmbedding("./data/embeddings/glove.840B.300d.txt")
# glove = GloVeEmbedding(file_path=None, dimensionality=100)
# model = LSTM(dataset, embedding=glove, save_tokenizer="./models/saved/tokenizer.json")
#
# # model.start_tuning()
# model.train()

# model = kNN(dataset)

# dataset = Dataset(csv_path="./data/saved/descriptions_25.csv", test_split=0.4, shuffle=True)


# incorrect_df = model.evaluate(load_model_path="./models/saved/lstm-small.h5", num_incorrect_examples="full")
# print(len(incorrect_df))
#
# incorrect_df = incorrect_df.loc[incorrect_df["probability"] > 0.8]
# for i, row in incorrect_df.iterrows():
#     print(row)
# print(len(incorrect_df))

if __name__ == "__main__":
    main()

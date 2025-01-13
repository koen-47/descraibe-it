"""
File to run main.py
"""

import argparse

import pandas as pd

from data.GloVeEmbedding import GloVeEmbedding
from data.Dataset import Dataset
from models.LSTM import LSTM
from models.kNN import kNN
from models.XGBoost import XGBoost
from models.SVM import SVM


def main():
    # Handle --model + --verbose argument.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    model_type = args.model
    verbose = args.verbose

    # Load training + test + validation data.
    train_data = pd.read_csv("./data/splits/train.csv")
    test_data = pd.read_csv("./data/splits/test.csv")
    val_data = pd.read_csv("./data/splits/val.csv")

    # Preprocess all data.
    pipeline = ["make_lowercase", "expand_contractions", "remove_stopwords", "clean_text"]
    dataset = Dataset(train_data=train_data, test_data=test_data, val_data=val_data, preprocess=pipeline)

    # Run kNN model
    if model_type == "knn":
        # Load best parameters for kNN model (see ./results/knn/knn_tuning.json)
        best_params = {
            "n_neighbors": 35,
            "p": 2,
            "weights": "distance"
        }

        model = kNN(dataset, best_params)
        
        print()

        # Perform 5-fold cross validation with the kNN model and get the best model across all splits
        best_accuracy, best_model, _ = model.cross_validate(5, verbose=verbose)
        print(f"Best accuracy: {best_accuracy}\n")

        # Evaluate the best kNN model on the test set
        best_model.evaluate(verbose=verbose)

    # Run XGBoost model
    elif model_type == "xgboost":
        # Load best parameters for XGBoost model (see ./results/xgboost/xgboost_tuning.json)
        best_params = {
            "learning_rate": 0.1,
            "max_depth": 7,
            "n_estimators": 1000
        }

        model = XGBoost(dataset, best_params)

        print()

        # Perform 5-fold cross validation with the XGBoost model and get the best model across all splits
        best_accuracy, best_model, _ = model.cross_validate(5, verbose=verbose)
        print(f"Best accuracy: {best_accuracy}\n")

        # Evaluate the best XGBoost model on the test set
        best_model.evaluate(verbose=verbose)

    # Run SVM model
    elif model_type == "svm":
        # Load best parameters for SVM model (see ./results/svm/svm_tuning.json)
        best_params = {
            "C": 10.0,
            "gamma": 1.0
        }

        model = SVM(dataset, best_params)

        print()

        # Perform 5-fold cross validation with the SVM model and get the best model across all splits
        best_accuracy, best_model, _ = model.cross_validate(5, verbose=verbose)
        print(f"Best accuracy: {best_accuracy}\n")

        # Evaluate the best SVM model on the test set
        best_model.evaluate(verbose=verbose)

    # Run LSTM model
    elif model_type == "lstm":
        # Load best parameters for LSTM model (see tuning.json files under ./results/lstm)
        best_params = {
            "lstm_layers": [{"bidirectional": True, "units": 448}],
            "fc_layers": [{"units": 384, "dropout_p": 0.7}],
            "early_stopping": {"patience": 20},
            "scheduler": {"initial_learning_rate": 0.001, "decay_steps": 25},
            "optimizer": {"name": "adam", "beta_1": 0.906, "beta_2": 0.955},
            "misc": {"epochs": 500, "batch_size": 256, "verbose": int(verbose)}
        }

        print()

        # Load the 840B + 300d pretrained GloVe word embeddings
        glove = GloVeEmbedding(f"./data/embeddings/glove.840B.300d.txt", dimensionality=300)
        model = LSTM(dataset, embedding=glove, params=best_params)

        print()

        # Perform 5-fold cross validation with the LSTM model and get the best model across all splits
        best_accuracy, best_model, _ = model.cross_validate(5, verbose=verbose)
        print(f"Best accuracy: {best_accuracy}\n")

        # Evaluate the best LSTM model on the test set
        best_model.evaluate(verbose=verbose)


if __name__ == "__main__":
    main()

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.simplefilter("default")
warnings.filterwarnings(
    "ignore",
    message=r"^non-integer arguments to randrange\(\).*deprecated since Python 3\.10.*$",
    category=DeprecationWarning
)

import io
import json
import math
from abc import ABC
import random
import functools

import optuna
import numpy as np
import scipy
import pickle
import tensorflow as tf
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TextVectorization, Bidirectional, Dropout, BatchNormalization, LeakyReLU
from keras.layers import Bidirectional, LayerNormalization
from tensorflow.keras.layers import LSTM as LSTMLayer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.saving import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from keras.layers import Embedding, GlobalMaxPooling1D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from data.GloVeEmbedding import GloVeEmbedding
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, \
    f1_score
import matplotlib.pyplot as plt

from models.Model import Model

# Set the seed for reproducibility
tf.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)



class LSTM(Model):
    def __init__(self, dataset, embedding, params, save_tokenizer=None):
        self.__dataset = dataset
        self.__train = dataset.train
        self.__val = None if dataset.val is None else dataset.val
        self.__test = dataset.test
        self.__embedding = embedding
        self.__tokenizer = Tokenizer()
        self.__tokenizer.fit_on_texts(self.__train["description"])
        self.__model = None
        self.__param_space = {}
        self.__params = params

        if save_tokenizer is not None:
            tokenizer_json = self.__tokenizer.to_json()
            with io.open(save_tokenizer, 'w+', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def fit(self, x_train=None, y_train=None, x_val=None, y_val=None, verbose=False):
        if x_train is None and y_train is None:
            x_train = self.__train["description"]
            y_train = self.__train["label"]

        if x_val is None and y_val is None:
            x_val = self.__val["description"]
            y_val = self.__val["label"]

        x_train = self.__tokenizer.texts_to_sequences(x_train)
        x_train = np.array(pad_sequences(x_train, maxlen=self.__embedding.dimensionality))
        y_train = np.array(y_train)
        vocab_size = len(self.__tokenizer.word_index) + 1
        embedding_matrix = self.__embedding.compute_embedding_matrix(self.__tokenizer, vocab_size)

        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))
        for lstm_layer in self.__params["lstm_layers"]:
            units = lstm_layer["units"]
            bidirectional = lstm_layer["bidirectional"]
            layer = Bidirectional(LSTMLayer(units)) if bidirectional else LSTMLayer(units)
            model.add(layer)
            model.add(LayerNormalization())
        for fc_layer in self.__params["fc_layers"]:
            units = fc_layer["units"]
            dropout_p = fc_layer["dropout_p"]
            model.add(Dense(units, activation="relu"))
            if dropout_p is not None:
                model.add(Dropout(dropout_p))
        model.add(Dense(25, activation='softmax'))

        initial_lr = self.__params["scheduler"]["initial_learning_rate"]
        decay_steps = self.__params["scheduler"]["decay_steps"]
        scheduler = CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps)
        scheduler = LearningRateScheduler(scheduler)

        optimizer = None
        if self.__params["optimizer"]["name"] == "adam":
            beta_1 = self.__params["optimizer"]["beta_1"]
            beta_2 = self.__params["optimizer"]["beta_2"]
            optimizer = Adam(beta_1=beta_1, beta_2=beta_2)
        elif self.__params["optimizer"]["name"] == "sgd":
            momentum = self.__params["optimizer"]["momentum"]
            optimizer = SGD(momentum=momentum)

        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=self.__params["misc"]["verbose"],
                           patience=self.__params["early_stopping"]["patience"])

        x_val = self.__tokenizer.texts_to_sequences(x_val)
        x_val = np.array(pad_sequences(x_val, maxlen=self.__embedding.dimensionality))
        y_val = np.array(y_val)
        val_history = model.fit(x_train, y_train, batch_size=self.__params["misc"]["batch_size"],
                                validation_data=(x_val, y_val), epochs=self.__params["misc"]["epochs"],
                                verbose=self.__params["misc"]["verbose"], callbacks=[es, scheduler])

        # model.save(self.__params["misc"]["save_filepath"], save_format="h5")
        self.__model = model
        return val_history

    def predict(self, x):
        x = self.__dataset.clean_text(x)
        x = self.__tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=self.__embedding.dimensionality)
        pred_probs = self.__model.predict(x)
        pred_max_prob = pred_probs.max(axis=-1)
        pred_label = self.__dataset.decode_label(pred_probs.argmax(axis=-1))
        return pred_label, pred_max_prob, pred_probs

    def evaluate(self, x_test=None, y_test=None, use_val=False, verbose=False):
        if x_test is None and y_test is None:
            y_test = self.__val["label"] if use_val else self.__test["label"]
            x_test = self.__val["description"] if use_val else self.__test["description"]

        label_encoder = self.__dataset.label_encoder
        x_test_tok = self.__tokenizer.texts_to_sequences(x_test)
        x_test_tok = pad_sequences(x_test_tok, maxlen=self.__embedding.dimensionality)

        y_pred = self.__model.predict(x_test_tok, verbose=0)
        y_pred_label = label_encoder.inverse_transform([np.argmax(pred) for pred in y_pred])
        y_test = label_encoder.inverse_transform(y_test)

        accuracy = float(accuracy_score(y_test, y_pred_label)) * 100
        precision = float(precision_score(y_test, y_pred_label, average='macro')) * 100
        recall = float(recall_score(y_test, y_pred_label, average='macro')) * 100
        f1 = float(f1_score(y_test, y_pred_label, average='macro')) * 100

        if verbose:
            print(f"Results for LSTM model:\n- Accuracy: {accuracy:.2f}%\n- Precision: {precision:.2f}%"
                  f"\n- Recall: {recall:.2f}%\n- F1 score: {f1:.2f}%")

        return accuracy, precision, recall, f1

    def plot_confusion_matrix(self):
        test_seq = self.__tokenizer.texts_to_sequences(self.__test["description"])
        test_seq = pad_sequences(test_seq, maxlen=self.__embedding.dimensionality)
        pred_labels = self.__model.predict(test_seq)
        pred_labels = [y.argmax(axis=-1) for y in pred_labels]
        labels = self.__dataset.decode_label(range(self.__test["label"].nunique()))
        cm_display = ConfusionMatrixDisplay.from_predictions(self.__test["label"], pred_labels,
                                                             normalize="true", display_labels=labels,
                                                             values_format=".2f", include_values=False)
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.17)
        plt.savefig("./visualizations/confusion_matrix_lstm.png", bbox_inches="tight")
        plt.show()

    def __create_model(self, trial):
        vocab_size = len(self.__tokenizer.word_index) + 1
        embedding_matrix = self.__embedding.compute_embedding_matrix(self.__tokenizer, vocab_size)

        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))

        params = {"architecture": {}, "optimizer": {"scheduler": {}, "adam": {}, "sgd": {}}}

        param_arch = self.__param_space["architecture"]
        if "n_lstm_units" in param_arch:
            n_lstm_units = param_arch["n_lstm_units"]
            n_lstm_units = trial.suggest_int("n_lstm_units", n_lstm_units["min"], n_lstm_units["max"],
                                             step=n_lstm_units["step"])
            params["architecture"]["n_lstm_units"] = n_lstm_units
            model.add(Bidirectional(LSTMLayer(n_lstm_units)))
        if "n_fc_layers" in param_arch:
            n_fc_layers = param_arch["n_fc_layers"]
            n_fc_layers = trial.suggest_int("n_fc_layers", n_fc_layers["min"], n_fc_layers["max"],
                                            step=n_fc_layers["step"])
            params["architecture"]["n_fc_layers"] = n_fc_layers
            n_fc_units = param_arch["n_fc_units"]
            n_fc_units = trial.suggest_int("n_fc_units", n_fc_units["min"], n_fc_units["max"],
                                           step=n_fc_units["step"])
            params["architecture"]["n_fc_units"] = n_fc_units
            dropout_p = param_arch["dropout_p"]
            dropout_prob = trial.suggest_float("dropout_prob", dropout_p["min"], dropout_p["max"],
                                               step=dropout_p["step"])
            params["architecture"]["dropout_p"] = dropout_prob
            for _ in range(n_fc_layers):
                model.add(Dense(n_fc_units, activation="relu"))
                model.add(Dropout(dropout_prob))

            model.add(Dense(25, activation='softmax'))

        optimizer = None
        param_optim = self.__param_space["optimizer"]
        lr_scheduler = 0.0001

        if "scheduler" in param_optim:
            param_scheduler = param_optim["scheduler"]
            initial_lr = trial.suggest_float("lr", param_scheduler["initial_lr"]["min"],
                                             param_scheduler["initial_lr"]["max"])
            decay_steps = trial.suggest_int("decay_steps", param_scheduler["decay_steps"]["min"],
                                            param_scheduler["decay_steps"]["max"],
                                            step=param_scheduler["decay_steps"]["step"])
            params["optimizer"]["scheduler"]["initial_lr"] = initial_lr
            params["optimizer"]["scheduler"]["decay_steps"] = decay_steps
            lr_scheduler = CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps)

        if "adam" in param_optim:
            param_adam = param_optim["adam"]
            beta_1 = trial.suggest_float("beta_1", param_adam["beta_1"]["min"], param_adam["beta_1"]["max"])
            beta_2 = trial.suggest_float("beta_2", param_adam["beta_2"]["min"], param_adam["beta_2"]["max"])
            optimizer = Adam(learning_rate=lr_scheduler, beta_1=beta_1, beta_2=beta_2)
            params["optimizer"]["adam"]["beta_1"] = beta_1
            params["optimizer"]["adam"]["beta_2"] = beta_2
        elif "sgd" in param_optim:
            param_sgd = param_optim["sgd"]
            momentum = trial.suggest_float("momentum", param_sgd["momentum"]["min"], param_sgd["momentum"]["max"])
            params["optimizer"]["sgd"]["momentum"] = momentum
            optimizer = SGD(learning_rate=lr_scheduler, momentum=momentum)

        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
        return model, params

    def __objective(self, trial, results, save=""):
        x_train = self.__tokenizer.texts_to_sequences(self.__train["description"])
        x_train = np.array(pad_sequences(x_train, maxlen=self.__embedding.dimensionality))
        y_train = np.array(self.__train["label"])

        x_val = self.__tokenizer.texts_to_sequences(self.__val["description"])
        x_val = np.array(pad_sequences(x_val, maxlen=self.__embedding.dimensionality))
        y_val = np.array(self.__val["label"])

        model, params = self.__create_model(trial)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
        model.fit(x_train, y_train, epochs=10000, batch_size=256,
                  validation_data=(x_train, y_train), verbose=0, callbacks=[es])
        score = model.evaluate(x_val, y_val)[0]

        results["trials"].append({"params": params, "score": score})
        if score < results["best_score"]:
            results["best_params"] = params
            results["best_score"] = score

        with open(save, "w") as file:
            json.dump(results, file, indent=3)

        return score

    def cross_validate(self, n_splits, verbose=False):
        cv = self.__dataset.get_cv_split(n_splits=n_splits, as_val=True)
        best_accuracy, best_model = 0, None
        results_per_split = []
        for i, data in enumerate(cv):
            x_train = data["train"]["description"]
            y_train = data["train"]["label"]
            x_test = data["test"]["description"]
            y_test = data["test"]["label"]
            model = LSTM(self.__dataset, embedding=self.__embedding, params=self.__params)
            loss_history = model.fit(x_train, y_train, x_test, y_test, verbose=False)
            accuracy, precision, recall, f1 = model.evaluate(x_test, y_test)
            results_per_split.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_loss_history": loss_history.history["loss"],
                "val_loss_history": loss_history.history["val_loss"]
            })
            if verbose:
                print(f"Accuracy on split {i+1}: {accuracy:.2f}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        return best_accuracy, best_model, results_per_split

    def tune(self, n_trials, param_space, save=""):
        self.__param_space = param_space
        study = optuna.create_study()
        results = {
            "param_space": param_space,
            "trials": [],
            "best_params": {},
            "best_score": math.inf
        }
        objective = functools.partial(self.__objective, results=results, save=save)
        study.optimize(objective, n_trials=n_trials)
        print(study.best_params)
        return study.best_params

# {'n_lstm_units': 368, 'n_fc_layers': 2, 'n_fc_width': 512, 'dropout_prob': 0.4}
# {'n_lstm_units': 480, 'n_fc_layers': 1, 'n_fc_width': 656, 'dropout_prob': 0.7000000000000001,
# 'lr': 0.0008575362871413658}

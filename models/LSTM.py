import io
import json
from abc import ABC
import random

import optuna
import numpy as np
import scipy
import pickle
import tensorflow
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TextVectorization, Bidirectional, Dropout, BatchNormalization, LeakyReLU
from keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM as LSTMLayer
from tensorflow.keras.optimizers import Adam
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
tensorflow.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)


class LSTM(Model):
    def __init__(self, dataset, embedding, save_tokenizer=None):
        self.__dataset = dataset
        self.__train = dataset.train
        self.__val = dataset.val
        self.__test = dataset.test
        self.__embedding = embedding
        self.__tokenizer = Tokenizer()
        self.__tokenizer.fit_on_texts(self.__train["description"])
        self.__model = None
        self.__hyperparameters = {}

        if save_tokenizer is not None:
            tokenizer_json = self.__tokenizer.to_json()
            with io.open(save_tokenizer, 'w+', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def fit(self, params):
        x_train = self.__tokenizer.texts_to_sequences(self.__train["description"])
        x_train = np.array(pad_sequences(x_train, maxlen=self.__embedding.dimensionality))
        y_train = np.array(self.__train["label"])
        vocab_size = len(self.__tokenizer.word_index) + 1
        embedding_matrix = self.__embedding.compute_embedding_matrix(self.__tokenizer, vocab_size)

        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))
        for lstm_layer in params["lstm_layers"]:
            units = lstm_layer["units"]
            bidirectional = lstm_layer["bidirectional"]
            layer = Bidirectional(LSTMLayer(units)) if bidirectional else LSTMLayer(units)
            model.add(layer)
        for fc_layer in params["fc_layers"]:
            units = fc_layer["units"]
            dropout_p = fc_layer["dropout_p"]
            model.add(Dense(units, activation="relu"))
            if dropout_p is not None:
                model.add(Dropout(dropout_p))
        model.add(Dense(25, activation='softmax'))
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["acc"])

        x_val = self.__tokenizer.texts_to_sequences(self.__val["description"])
        x_val = np.array(pad_sequences(x_val, maxlen=self.__embedding.dimensionality))
        y_val = np.array(self.__val["label"])
        scheduler = self.__get_lr_scheduler(params["scheduler"])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=params["early_stopping"]["verbose"],
                           patience=params["early_stopping"]["patience"])
        model.fit(x_train, y_train, batch_size=params["misc"]["batch_size"], validation_data=(x_val, y_val),
                  epochs=params["misc"]["epochs"], verbose=1,
                  callbacks=[es, scheduler])
        model.save(params["misc"]["save_filepath"], save_format="h5")
        self.__model = model

    def __get_lr_scheduler(self, params):
        initial_lr = params["initial_learning_rate"]
        decay_steps = params["decay_steps"]
        scheduler = CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps)
        scheduler = LearningRateScheduler(scheduler)
        return scheduler

    def predict(self, x):
        x = self.__dataset.clean_text(x)
        x = self.__tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=self.__embedding.dimensionality)
        pred_probs = self.__model.predict(x)
        pred_max_prob = pred_probs.max(axis=-1)
        pred_label = self.__dataset.decode_label(pred_probs.argmax(axis=-1))
        return pred_label, pred_max_prob, pred_probs

    def evaluate(self, get_misclassifications=True):
        label_encoder = self.__dataset.label_encoder
        x_test = self.__tokenizer.texts_to_sequences(self.__test["description"])
        x_test = pad_sequences(x_test, maxlen=self.__embedding.dimensionality)
        y_pred = self.__model.predict(x_test)

        x_test = self.__test["description"]
        y_pred_label = label_encoder.inverse_transform([np.argmax(pred) for pred in y_pred])
        y_pred_prob = [np.max(pred) for pred in y_pred]
        y_test = label_encoder.inverse_transform(self.__test["label"])

        accuracy = accuracy_score(y_test, y_pred_label)
        precision = precision_score(y_test, y_pred_label, average='micro')
        recall = recall_score(y_test, y_pred_label, average='micro')
        f1 = f1_score(y_test, y_pred_label, average='micro')

        if get_misclassifications:
            incorrect = [(desc, pred, prob, actual) for desc, pred, prob, actual in
                         zip(x_test, y_pred_label, y_pred_prob, y_test) if pred != actual]
            incorrect_df = pd.DataFrame(incorrect, columns=["description", "predicted", "probability", "actual"])
            return accuracy, precision, recall, f1, incorrect_df
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

        n_lstm_units = self.__hyperparameters["n_lstm_units"]
        n_fc_layers = self.__hyperparameters["n_fc_layers"]
        n_fc_units = self.__hyperparameters["n_fc_units"]
        dropout_p = self.__hyperparameters["dropout_p"]

        n_lstm_units = trial.suggest_int("n_lstm_units", n_lstm_units["min"], n_lstm_units["max"],
                                         step=n_lstm_units["step"])
        n_fc_layers = trial.suggest_int("n_fc_layers", n_fc_layers["min"], n_fc_layers["max"], step=n_fc_layers["step"])
        n_fc_units = trial.suggest_int("n_fc_units", n_fc_units["min"], n_fc_units["max"], step=n_fc_units["step"])
        dropout_prob = trial.suggest_float("dropout_prob", dropout_p["min"], dropout_p["max"], step=dropout_p["step"])

        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))
        model.add(Bidirectional(LSTMLayer(n_lstm_units)))
        for _ in range(n_fc_layers):
            model.add(Dense(n_fc_units, activation="relu"))
            model.add(Dropout(dropout_prob))
        model.add(Dense(25, activation='softmax'))
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["acc"])
        return model

    def __objective(self, trial):
        x_train = self.__tokenizer.texts_to_sequences(self.__train["description"])
        x_train = np.array(pad_sequences(x_train, maxlen=self.__embedding.dimensionality))
        y_train = np.array(self.__train["label"])
        x_val = self.__tokenizer.texts_to_sequences(self.__val["description"])
        x_val = np.array(pad_sequences(x_val, maxlen=self.__embedding.dimensionality))
        y_val = np.array(self.__val["label"])
        model = self.__create_model(trial)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
        model.fit(x_train, y_train, epochs=10000, batch_size=256,
                  validation_data=(x_train, y_train), verbose=0, callbacks=[es])
        score = model.evaluate(x_val, y_val)
        return score[0]

    def tune(self, n_trials, hyperparameters):
        self.__hyperparameters = hyperparameters
        study = optuna.create_study()
        study.optimize(self.__objective, n_trials=n_trials)
        print(study.best_params)
        return study.best_params

# {'n_lstm_units': 368, 'n_fc_layers': 2, 'n_fc_width': 512, 'dropout_prob': 0.4}
# {'n_lstm_units': 480, 'n_fc_layers': 1, 'n_fc_width': 656, 'dropout_prob': 0.7000000000000001,
# 'lr': 0.0008575362871413658}

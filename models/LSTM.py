import io
import json
from abc import ABC
import random

import optuna
import numpy as np
import scipy
import pickle
import tensorflow
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TextVectorization, Bidirectional, Dropout, BatchNormalization, LeakyReLU
from keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM as LSTMLayer
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.saving import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from keras.layers import Embedding, GlobalMaxPooling1D
from sklearn.feature_extraction.text import TfidfVectorizer
from data.GloVeEmbedding import GloVeEmbedding
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Set the seed for reproducibility
tensorflow.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)


class LSTM:
    """
    Class in which the LSTM model is implemented.
    """

    def __init__(self, dataset, embedding, load_model_path=None, save_tokenizer=None):
        """
        Constructor for the LSTM class.
        :param dataset: Dataset object that contains all the necessary, correctly formatted data.
        :param load_model_path: String that contains the path to load a pretrained model.
        :param save_tokenizer: String that contains the path to show where to save tokenizer that has been fitted to
        the training data.
        """
        self.__dataset = dataset
        self.__train = dataset.train
        self.__test = dataset.test
        self.__embedding = embedding
        self.__tokenizer = Tokenizer()
        self.__tokenizer.fit_on_texts(self.__train["description"])
        self.__model = None

        if save_tokenizer is not None:
            tokenizer_json = self.__tokenizer.to_json()
            with io.open(save_tokenizer, 'w+', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def train(self):
        """
        Class function to start training the model.
        :param embedding: Embedding object that contains all the necessary information related to the embedding (i.e.,
        embedding matrix and word index)
        """
        train_seq = self.__tokenizer.texts_to_sequences(self.__train["description"])
        train_seq = pad_sequences(train_seq, maxlen=self.__embedding.dimensionality)
        vocab_size = len(self.__tokenizer.word_index) + 1
        embedding_matrix = self.__embedding.compute_embedding_matrix(self.__tokenizer, vocab_size)

        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))
        model.add(Bidirectional(LSTMLayer(480)))
        model.add(Dense(656, activation="relu"))
        model.add(Dropout(0.7))
        model.add(Dense(25, activation='softmax'))
        optimizer = Adam(learning_rate=0.0085)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["acc"])
        print(model.summary())

        # {'n_lstm_units': 368, 'n_fc_layers': 2, 'n_fc_width': 512, 'dropout_prob': 0.4}
        # {'n_lstm_units': 480, 'n_fc_layers': 1, 'n_fc_width': 656, 'dropout_prob': 0.7000000000000001, 'lr': 0.0008575362871413658}

        scheduler = tensorflow.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=50)
        scheduler = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        model.fit(np.array(train_seq), np.array(self.__train["label"]), batch_size=64,
                  validation_split=0.2, epochs=10000, verbose=1, callbacks=[es, scheduler])
        model.save("./models/saved/lstm.h5", save_format="h5")

    def predict(self, text):
        """
        Class function to predict a word based on the given text. This function cleans and tokenizes the raw input text.
        :param text: Raw input text.
        :return: Returns the predicted label (string), its probability (float) and the probabilities for all classes/
        words (array of floats).
        """
        text = self.__dataset.clean_text(text)
        text = self.__tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=self.__embedding.dimensionality)
        pred_probs = self.__model.predict(text)
        pred_max_prob = pred_probs.max(axis=-1)
        pred_label = self.__dataset.decode_label(pred_probs.argmax(axis=-1))
        return pred_label, pred_max_prob, pred_probs

    def evaluate(self, load_model_path=None, num_incorrect_examples=10):
        """
        Evaluates the model on the test set based on the accuracy.
        :return: Returns the score containing test set accuracy and loss.
        """
        if load_model_path is not None:
            self.__model = load_model(load_model_path)
        test_seq = self.__tokenizer.texts_to_sequences(self.__test["description"])
        test_seq = pad_sequences(test_seq, maxlen=self.__embedding.dimensionality)
        score = self.__model.evaluate(test_seq, self.__test["label"], verbose=0)
        print(f"Test set loss: {score[0]}")
        print(f"Test set accuracy: {score[1]}")
        return score

    def plot_confusion_matrix(self):
        """
        Generates a plot containing a confusion matrix with all classes.
        """
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

        n_lstm_units = trial.suggest_int('n_lstm_units', 16, 512, step=16)
        n_fc_layers = trial.suggest_int('n_fc_layers', 1, 10, step=1)
        n_fc_width = trial.suggest_int('n_fc_width', 16, 2048, step=64)
        dropout_prob = trial.suggest_float('dropout_prob', 0., 0.9, step=0.1)
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))
        model.add(Bidirectional(LSTMLayer(n_lstm_units)))
        for _ in range(n_fc_layers):
            model.add(Dense(n_fc_width, activation="relu"))
            model.add(Dropout(dropout_prob))
        model.add(Dense(25, activation='softmax'))
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["acc"])
        return model

    def objective(self, trial):
        train_seq = self.__tokenizer.texts_to_sequences(self.__train["description"])
        train_seq = pad_sequences(train_seq, maxlen=self.__embedding.dimensionality)
        model = self.__create_model(trial)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
        model.fit(np.array(train_seq), np.array(self.__train["label"]), epochs=10000, batch_size=512,
                  validation_split=0.2, verbose=0, callbacks=[es])
        test_seq = self.__tokenizer.texts_to_sequences(self.__test["description"])
        test_seq = pad_sequences(test_seq, maxlen=self.__embedding.dimensionality)
        score = model.evaluate(np.array(test_seq), np.array(self.__test["label"]))
        return score[0]

    def start_tuning(self):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=100)
        print(study.best_params)

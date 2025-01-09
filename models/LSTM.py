"""
File containing all functionality related to the LSTM model.
"""

import os
import warnings

# Suppress information printed by Tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import io
import json
import math
import random
import functools

import optuna
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import Bidirectional, LayerNormalization
from tensorflow.keras.layers import LSTM as LSTMLayer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.layers import Embedding
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.Model import Model
from data.PreprocessingPipeline import PreprocessingPipeline

# Set the seed for reproducibility
tf.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)


class LSTM(Model):
    """
    Class to implement functionality for the LSTM model.
    """
    def __init__(self, dataset, embedding, params, save_tokenizer=None):
        """
        Constructor for the LSTM model.
        :param dataset: Dataset instance containing the whole dataset to run the LSTM on.
        :param embedding: GloVeEmbedding instance to handle the pretrained GloVe embeddings.
        :param params: dictionary determining the parameters of the LSTM model.
        :param save_tokenizer: path to save the tokenizer to (used for the demo).
        """

        # Parameters for the dataset.
        self.__dataset = dataset
        self.__train = dataset.train
        self.__val = None if dataset.val is None else dataset.val
        self.__test = dataset.test

        # Parameters for the model.
        self.__model = None
        self.__param_space = {}
        self.__params = params
        self.__embedding = embedding

        # Create tokenizer and fit it to the training set of the dataset.
        self.__tokenizer = Tokenizer()
        self.__tokenizer.fit_on_texts(self.__train["description"])

        # Save the tokenizer as a JSON file (if necessary).
        if save_tokenizer is not None:
            tokenizer_json = self.__tokenizer.to_json()
            with io.open(save_tokenizer, 'w+', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def fit(self, x_train=None, y_train=None, x_val=None, y_val=None):
        """
        Fits the LSTM to a (specified) dataset.
        :param x_train: features of the training set. If not specified, it will default to the training features of the
        dataset specified during instantiation.
        :param y_train: labels of the training set. If not specified, it will default to the training features of the
        dataset specified during instantiation.
        :param x_val: features of the validation set. If not specified, it will default to the validation features of
        the dataset specified during instantiation.
        :param y_val: labels of the validation set. If not specified, it will default to the validation features of
        the dataset specified during instantiation.
        :return: history containing the training + validation loss per epoch.
        """

        # Get the training data from self.dataset if it is not explicitly specified.
        if x_train is None and y_train is None:
            x_train = self.__train["description"]
            y_train = self.__train["label"]

        # Get the validation data from self.dataset if it is not explicitly specified.
        if x_val is None and y_val is None:
            x_val = self.__val["description"]
            y_val = self.__val["label"]

        # Convert training data using tokenizer and sequence padding
        x_train = self.__tokenizer.texts_to_sequences(x_train)
        x_train = np.array(pad_sequences(x_train, maxlen=self.__embedding.dimensionality))
        y_train = np.array(y_train)

        # Set up the pretrained GloVe word embeddings
        vocab_size = len(self.__tokenizer.word_index) + 1
        embedding_matrix = self.__embedding.compute_embedding_matrix(self.__tokenizer, vocab_size)

        # Set up model and add embedding layer
        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))

        # Add LSTM layer
        for lstm_layer in self.__params["lstm_layers"]:
            units = lstm_layer["units"]
            bidirectional = lstm_layer["bidirectional"]
            layer = Bidirectional(LSTMLayer(units)) if bidirectional else LSTMLayer(units)
            model.add(layer)
            model.add(LayerNormalization())

        # Add fully-connected layers
        for fc_layer in self.__params["fc_layers"]:
            units = fc_layer["units"]
            dropout_p = fc_layer["dropout_p"]
            model.add(Dense(units, activation="relu"))
            if dropout_p is not None:
                model.add(Dropout(dropout_p))
        model.add(Dense(25, activation='softmax'))

        # Set up learning rate scheduler (Cosine Decay)
        initial_lr = self.__params["scheduler"]["initial_learning_rate"]
        decay_steps = self.__params["scheduler"]["decay_steps"]
        scheduler = CosineDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps)
        scheduler = LearningRateScheduler(scheduler)

        # Set up optimizer (Adam or SGD)
        optimizer = None
        if self.__params["optimizer"]["name"] == "adam":
            beta_1 = self.__params["optimizer"]["beta_1"]
            beta_2 = self.__params["optimizer"]["beta_2"]
            optimizer = Adam(beta_1=beta_1, beta_2=beta_2)
        elif self.__params["optimizer"]["name"] == "sgd":
            momentum = self.__params["optimizer"]["momentum"]
            optimizer = SGD(momentum=momentum)

        # Compile model with sparse cross-entropy as the loss function + early stopping
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=self.__params["misc"]["verbose"],
                           patience=self.__params["early_stopping"]["patience"])

        # Convert training data using tokenizer and sequence padding
        x_val = self.__tokenizer.texts_to_sequences(x_val)
        x_val = np.array(pad_sequences(x_val, maxlen=self.__embedding.dimensionality))
        y_val = np.array(y_val)

        # Fit the model to training data and evaluate on validation data
        history = model.fit(x_train, y_train, batch_size=self.__params["misc"]["batch_size"],
                            validation_data=(x_val, y_val), epochs=self.__params["misc"]["epochs"],
                            verbose=self.__params["misc"]["verbose"], callbacks=[es, scheduler])
        self.__model = model

        # Return the history of the training + validation loss/accuracy.
        return history

    def predict(self, x):
        """
        Perform a prediction on the specified data points.
        :param x: specified data point to perform the prediction on.
        :return: a triple consisting of the predicted label, the probability of that prediction, and a list containing
        the probabilities of all labels.
        """

        # Preprocess the specified data points.
        x = PreprocessingPipeline(self.__dataset.preprocess).apply(x)
        x = self.__tokenizer.texts_to_sequences(x)
        x = pad_sequences(x, maxlen=self.__embedding.dimensionality)

        # Perform prediction
        pred_probs = self.__model.predict(x)
        pred_max_prob = pred_probs.max(axis=-1)
        pred_label = self.__dataset.decode_label(pred_probs.argmax(axis=-1))
        return pred_label, pred_max_prob, pred_probs

    def evaluate(self, x_test=None, y_test=None, use_val=False, verbose=False):
        """
        Evaluate the fitted model on the test or validation set.
        :param x_test: specified test/validation features to evaluate on.
        :param y_test: specified test/validation labels to evaluate on.
        :param use_val: flag to indicate if the model should be evaluated on the test set or validation set.
        :param verbose: flag to indicate if the results should be printed at the end.
        :return: quadruple consisting of the accuracy, precision, recall and f1-score.
        """

        # Get either the validation or test set if they are specified.
        if x_test is None and y_test is None:
            y_test = self.__val["label"] if use_val else self.__test["label"]
            x_test = self.__val["description"] if use_val else self.__test["description"]

        # Convert test/validation data using tokenizer and sequence padding
        x_test = self.__tokenizer.texts_to_sequences(x_test)
        x_test = pad_sequences(x_test, maxlen=self.__embedding.dimensionality)
        y_pred = self.__model.predict(x_test, verbose=0)

        # Decode the label from its number to string.
        label_encoder = self.__dataset.label_encoder
        y_pred_label = label_encoder.inverse_transform([np.argmax(pred) for pred in y_pred])
        y_test = label_encoder.inverse_transform(y_test)

        # Compute evaluation metrics.
        accuracy = float(accuracy_score(y_test, y_pred_label)) * 100
        precision = float(precision_score(y_test, y_pred_label, average='macro')) * 100
        recall = float(recall_score(y_test, y_pred_label, average='macro')) * 100
        f1 = float(f1_score(y_test, y_pred_label, average='macro')) * 100

        # Print metric results if verbosity is set.
        if verbose:
            print(f"Results for LSTM model:\n- Accuracy: {accuracy:.2f}%\n- Precision: {precision:.2f}%"
                  f"\n- Recall: {recall:.2f}%\n- F1 score: {f1:.2f}%")

        return accuracy, precision, recall, f1

    def cross_validate(self, n_splits, verbose=False):
        """
        Perform cross validation on self.dataset based on the specified number of splits to compute the best performing
        model.
        :param n_splits: specified number of splits to perform cross validation on.
        :param verbose: flag to denote if results per split are printed.
        :return: triple consisting of the best accuracy of the best performing model, the best performing model,
        and the results per split (accuracy, precision, recall, f1-score, train + validation loss/accuracy per epoch).
        """

        # Concatenate the training + validation sets and split them into the specified number of splits.
        cv = self.__dataset.get_cv_split(n_splits=n_splits, as_val=True)
        best_accuracy, best_model = 0, None
        results_per_split = []

        # Iterate over each split.
        for i, data in enumerate(cv):
            # Get training + validation set of that split.
            x_train = data["train"]["description"]
            y_train = data["train"]["label"]
            x_test = data["test"]["description"]
            y_test = data["test"]["label"]

            # Fit the model to training set and evaluate on the validation set.
            model = LSTM(self.__dataset, embedding=self.__embedding, params=self.__params)
            loss_history = model.fit(x_train, y_train, x_test, y_test)
            loss_history = loss_history.history
            accuracy, precision, recall, f1 = model.evaluate(x_test, y_test)

            # Record results for that split.
            results_per_split.append({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_loss_history": loss_history["loss"],
                "val_loss_history": loss_history["val_loss"],
                "train_acc_history": loss_history["acc"],
                "val_acc_history": loss_history["val_acc"]
            })

            # Print results if verbosity is set.
            if verbose:
                print(f"Accuracy on split {i + 1}: {accuracy:.2f}")

            # Record best performing model and accuracy of that model.
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        return best_accuracy, best_model, results_per_split

    def __create_model(self, trial):
        """
        Create the model that will be evaluated using Bayesian optimization.
        :param trial: current trial that is being performed.
        :return: tuple consisting of the created model and the parameters of that model.
        """

        # Set up pretrained GloVe word embeddings
        vocab_size = len(self.__tokenizer.word_index) + 1
        embedding_matrix = self.__embedding.compute_embedding_matrix(self.__tokenizer, vocab_size)

        # Set up model and add embedding layer
        model = Sequential()
        model.add(Embedding(vocab_size, self.__embedding.dimensionality, weights=[embedding_matrix],
                            input_length=self.__embedding.dimensionality, trainable=False))

        # Set up dictionary to record which parameters are being set.
        params = {"architecture": {}, "optimizer": {"scheduler": {}, "adam": {}, "sgd": {}}}

        # Set up the architecture of the LSTM model.
        param_arch = self.__param_space["architecture"]

        # Add LSTM layer
        if "n_lstm_units" in param_arch:
            n_lstm_units = param_arch["n_lstm_units"]
            n_lstm_units = trial.suggest_int("n_lstm_units", n_lstm_units["min"], n_lstm_units["max"],
                                             step=n_lstm_units["step"])
            params["architecture"]["n_lstm_units"] = n_lstm_units
            model.add(Bidirectional(LSTMLayer(n_lstm_units)))

        # Add fully-connected layers
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

        # Set up optimizer
        optimizer = None
        param_optim = self.__param_space["optimizer"]
        lr_scheduler = 0.0001

        # Set up learning rate scheduler (Cosine Decay)
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

        # Set up optimizer (Adam or SGD)
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

        # Compile model with sparse cross-entropy as the loss function
        model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["acc"])
        return model, params

    def __objective(self, trial, results, save=""):
        """
        Objective function to optimize during Bayesian optimization.
        :param trial: current trial that is being performed.
        :param results: dictionary containing results for each trial performed so far.
        :param save: path of location where to save the results to.
        :return:
        """

        # Convert training data using tokenizer and sequence padding
        x_train = self.__tokenizer.texts_to_sequences(self.__train["description"])
        x_train = np.array(pad_sequences(x_train, maxlen=self.__embedding.dimensionality))
        y_train = np.array(self.__train["label"])

        # Convert validation data using tokenizer and sequence padding
        x_val = self.__tokenizer.texts_to_sequences(self.__val["description"])
        x_val = np.array(pad_sequences(x_val, maxlen=self.__embedding.dimensionality))
        y_val = np.array(self.__val["label"])

        # Create model, fit it to the training data, and evaluate it on the validation data.
        model, params = self.__create_model(trial)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
        model.fit(x_train, y_train, epochs=10000, batch_size=256,
                  validation_data=(x_train, y_train), verbose=0, callbacks=[es])
        score = model.evaluate(x_val, y_val)[0]

        # Record results and save them.
        results["trials"].append({"params": params, "score": score})
        if score < results["best_score"]:
            results["best_params"] = params
            results["best_score"] = score

        with open(save, "w") as file:
            json.dump(results, file, indent=3)

        return score

    def tune(self, n_trials, param_space, save=""):
        """
        Perform hyperparameter tuning through Bayesian optimization, based on the specified number of trials and the
        parameter space.
        :param n_trials: number of trials of Bayesian optimization to perform.
        :param param_space: parameter space to search over.
        :param save: path of location where to save the results to.
        :return: the best performing parameters after tuning is complete.
        """

        self.__param_space = param_space
        study = optuna.create_study()
        results = {
            "param_space": param_space,
            "trials": [],
            "best_params": {},
            "best_score": math.inf
        }

        # Create helper objective function to help with recording results.
        objective = functools.partial(self.__objective, results=results, save=save)

        # Start Bayesian optimization.
        study.optimize(objective, n_trials=n_trials)
        print(study.best_params)
        return study.best_params


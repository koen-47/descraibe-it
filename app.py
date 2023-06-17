import json
import random
import re
import time

import flask
import keras_preprocessing.text
import numpy as np
import tensorflow
from flask import Flask, request
from flask_cors import CORS, cross_origin
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

"""
File where all the code related to the Flask server is handled.
"""

tensorflow.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# def load_resources():
#     global label_encoder, model
#     with open('./models/saved/tokenizer.json') as f:
#         data = json.load(f)
#         global tokenizer
#         tokenizer = keras_preprocessing.text.tokenizer_from_json(data)
#     label_encoder = LabelEncoder()
#     label_encoder.classes_ = np.load("./models/saved/labels.npy", allow_pickle=True)
#     model = keras.models.load_model("./models/saved/lstm-small.h5")
#
#     print(model)
#     print(label_encoder)
#     print(tokenizer)
#
#
# load_resources()
# print("All resources loaded...")


@app.route("/predict", methods=["POST"])
@cross_origin()
def index():
    """
    Creates a /predict route for the Flask API that handles the model predictions (in the form of POST requests).
    It expects the raw input text to be sent with the request.
    :return: Returns a response containing the predictions from the model based on the raw input text that was sent
    with the request.
    """
    print("Received request...")

    start_time = time.time()
    print(f"Start: {time.time() - start_time}")

    model = keras.models.load_model("./models/saved/lstm.h5")
    with open('./models/saved/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = keras_preprocessing.text.tokenizer_from_json(data)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("./models/saved/labels.npy", allow_pickle=True)

    input_json = request.get_json(force=True)
    text = [clean_text(input_json["text"])]
    text = tokenizer.texts_to_sequences(text)
    text = keras.preprocessing.sequence.pad_sequences(text, maxlen=100)
    print(f"Preprocessing: {time.time() - start_time}")

    pred_label = model.predict(text)[0]
    print(f"Predicting: {time.time() - start_time}")

    response_dict = {}
    for i in range(len(pred_label)):
        label = label_encoder.inverse_transform([i]).tolist()[0]
        response_dict[label] = float(pred_label[i])
    response_dict = dict(sorted(response_dict.items(), key=lambda item: item[1], reverse=True))
    print(response_dict)
    print(f"Response: {time.time() - start_time}")
    response = flask.jsonify(response_dict)
    return response


def clean_text(text):
    """
    Miscellaneous function to handle text cleaning (it is not possible to import the Dataset class into this file).
    :param text: The raw input text to be cleaned.
    :return: Returns the cleaned text.
    """
    text = text.lower()
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'^(\d{1,2})(.|\)) ', '', text)
    text = re.sub(r'  ', ' ', text)
    return text

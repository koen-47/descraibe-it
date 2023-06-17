from django.shortcuts import render

import json
import random
import re
import time

import keras_preprocessing.text
import numpy as np
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder



def index(request):
    context = {
        "prediction": {
            "word": "",
            "probability": -1
        }
    }

    if request.GET.get("btnSubmit"):
        try:
            model = keras.models.load_model("./models/saved/lstm-small.h5")
            with open('./models/saved/tokenizer.json') as f:
                data = json.load(f)
                tokenizer = keras_preprocessing.text.tokenizer_from_json(data)
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.load("./models/saved/labels.npy", allow_pickle=True)
        except IOError:
            model = keras.models.load_model("./saved/lstm-small.h5")
            with open('./saved/tokenizer.json') as f:
                data = json.load(f)
                tokenizer = keras_preprocessing.text.tokenizer_from_json(data)
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.load("./saved/labels.npy", allow_pickle=True)

        description = request.GET.get("txtDescription")
        text = [clean_text(description)]
        text = tokenizer.texts_to_sequences(text)
        text = keras.preprocessing.sequence.pad_sequences(text, maxlen=100)
        pred_label = model.predict(text)[0]
        response_dict = {}
        for i in range(len(pred_label)):
            label = label_encoder.inverse_transform([i]).tolist()[0]
            response_dict[label] = float(pred_label[i])
        response_dict = dict(sorted(response_dict.items(), key=lambda item: item[1], reverse=True))
        most_probable_word = next(iter(response_dict.items()))
        context["prediction"]["word"] = most_probable_word[0]
        context["prediction"]["probability"] = most_probable_word[1]

    return render(request, "index.html", context=context)


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

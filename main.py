import os

from sklearn.model_selection import KFold
from tabulate import tabulate
import pandas as pd

from data.GloVeEmbedding import GloVeEmbedding
from data.PromptManager import PromptManager
from data.Dataset import Dataset
from models.LSTM import LSTM
from models.SVM import SVM
from models.kNN import kNN
from models.RandomForest import IF
from models.LOF import LOF

args = {
    "prompt_template": 'Give me <var1> <var2>unique descriptions of <var3>. Do not include the word '
                       '"<var4>" or any of its variations in your response. Use <var5> language in your response.'
                       '<var6>',
    "length": [20],
    "detail": ["very short", "short", "", "long", "very long"],
    "complexity": ["very simple", "simple", "complex", "very complex"],
    "prefix": ["it", "this", "a", "the", "with", ""],
    "temperature": [0.2, 0.6, 1.0],
    "categories_file": "./data/saved/categories_25.txt"
}


def start_prompts():
    """
    This function starts prompting ChatGPT with the OpenAI API key (stored as an environment variable) and the arguments
    shown above/
    """
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    manager.start_prompts()


dataset = Dataset(csv_path="./data/saved/descriptions_25.csv", test_split=0.4, val_split=0.2, shuffle=True)
# glove = GloVeEmbedding("./data/embeddings/glove.6B.100d.txt", dimensionality=100)
# glove = GloVeEmbedding("./data/embeddings/glove.840B.300d.txt")
# glove = GloVeEmbedding(file_path=None, dimensionality=100)
# model = LSTM(dataset, embedding=glove, save_tokenizer="./models/saved/tokenizer.json")
#
# # model.start_tuning()
# model.train()

model = kNN(dataset)

# dataset = Dataset(csv_path="./data/saved/descriptions_25.csv", test_split=0.4, shuffle=True)



# incorrect_df = model.evaluate(load_model_path="./models/saved/lstm-small.h5", num_incorrect_examples="full")
# print(len(incorrect_df))
#
# incorrect_df = incorrect_df.loc[incorrect_df["probability"] > 0.8]
# for i, row in incorrect_df.iterrows():
#     print(row)
# print(len(incorrect_df))
import os

from data.GloVeEmbedding import GloVeEmbedding
from data.PromptManager import PromptManager
from data.Dataset import Dataset
from models.LSTM import LSTM
from models.SVM import SVM
from models.kNN import kNN

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


glove = GloVeEmbedding("./data/embeddings/glove.6B.100d.txt")
# glove = GloVeEmbedding("./data/embeddings/glove.840B.300d.txt")
dataset = Dataset(csv_path="./data/saved/descriptions_25.csv", test_split=0.4, shuffle=True)
model = LSTM(dataset, embedding=glove, save_tokenizer="./models/saved/tokenizer.json")
model.start_tuning()

# model.train()
# model.evaluate(load_model_path="./models/saved/lstm.h5")
# model.plot_confusion_matrix()

# descrAIbe it (NLP Assignment 3)
The web app [descrAIbe it](https://descraibe.onrender.com/) it is based on a game in which players are given a word and are tasked with describing
it as best they can without using the word itself. A player can score points if they get it right, otherwise their
score is reset back to zero and they must start over. Every round consists of a word being randomly sampled
from a pool of 100 unique words. For every word description a player submits, they are also given the class
probabilities associated with their answer in descending order.

For more information, please check the [report](https://github.com/Koen-Kraaijveld/nlp-assignment-3/blob/main/report.pdf).

## Directories
* [app.py](https://github.com/Koen-Kraaijveld/nlp-assignment-3/blob/main/app.py)
  * File that contains the Flask API and the routes that may be accessed.
* [./models](https://github.com/Koen-Kraaijveld/nlp-assignment-3/tree/main/models)
  * Directory that contains the LSTM model class and all saved files that are relevant to loading the model.
* [./data](https://github.com/Koen-Kraaijveld/nlp-assignment-3/tree/main/data)
  * Directory that contains all the functionality related to the data, including classes for data handling, word embeddings and data collection through ChatGPT prompting.
* [./web](https://github.com/Koen-Kraaijveld/nlp-assignment-3/tree/main/web)
  * Directory that contains the frontend React Typescript web app.

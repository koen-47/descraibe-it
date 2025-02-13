<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="./data/resources/descraibe-it-header-dark.png">
    <img alt="" src="./data/resources/descraibe-it-header-light.png" />
</picture>

---

Inspired by [Quick, Draw!](https://quickdraw.withgoogle.com/), *DescrAIbe It* is a text-based alternative where players must describe words for a trained NLP model to guess.
This is done by collecting a dataset of over 180,000 descriptions spread over 25 words.

[**🤗 Dataset**](https://huggingface.co/datasets/koen-47/descraibe-it) | [**🎮 Demo**](https://descraibe-it.onrender.com/)

</div>



## Installation

Create a conda environment with a name of your choice with Python version 3.10:


```shell
conda create -n [env_name] python=3.10
```

Activate it and install all necessary libraries:

```shell
pip install -r requirements.txt
```


### Reproducing the Results

Run the following commands to run a specified model on the data (for the setup used here, see the [evaluation](#evaluation) section).

```shell
python main.py --model [model] [verbosity]
```

The model arguments are: `knn`, `xgboost`, `svm` or `lstm`. To print the results, add `--verbose` to the end of the command.

In order to run the LSTM model, you need to download the [pretrained GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) (840B tokens + 300d vectors) and add them to the folder `data/embeddings`.


## Methodology

The following section offers a concise overview of the methodology used 
in the development of the dataset and the model experiments 
conducted on it. It includes links to the relevant code and 
results for reference. While it highlights the key aspects of the 
approach, it is not intended to be exhaustive (please refer to the 
code for a complete and detailed overview).

### Data Overview
The dataset is created through three key phases: selection, collection, and preparation. Each phase is detailed below 
to provide a clear understanding of the data pipeline. All steps were carried out in July 2023.

#### Data Selection
The first step towards collecting the dataset consists of selecting which words will 
need to be described by the player. For this, I take inspiration from the game [Quick, Draw!](https://quickdraw.withgoogle.com/), 
a computer vision alternative which uses doodles instead of text descriptions. 
First, I select the [words](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt) used in that game.
As my approach to word selection is based on word embeddings, I remove all [entries](./data/saved/categories_289.txt) consisting of more than one 
word to ensure I can compare the embeddings with each other.

From these remaining words, I further narrow them down to a subset of 25. This is done in the interest of 
the time and resource costs incurred by using OpenAI's API for large scale purposes, as many descriptions per word will be 
needed to accurately and fairly evaluate the validity of the more data hungry neural models. 

To select a suitable set of 25 words, I aim to create as semantically a diverse subset as possible.
This can be achieved by maximizing the minimum distance between the embeddings associated with each of the selected words.
More formally, for a set of words $W$, I want to find the subset of words $W' \subset W$ using the Euclidean
distance function $d(e(w_i), e(w_j))$, where $e(w)$ returns the embedding associated with word $w \in W$.
The subset $W'$ should satisfy the following:

$$
\mathop{\max} \left( \mathop{\min}\limits_{w_i, w_j \in W' \atop i \neq j} d \left( e(w_i), e(w_j) \right) \right)
$$

This is known as the [<em>max-min diversity problem</em>](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=51f97d822ec695f5700ac353bfff285bd44ef0e7) and is considered NP-hard. 
As such, I use a greedy algorithm that approximates the optimal subset. The embeddings I use are [pretrained GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) (840B tokens + 300d vectors).

The [final set of selected words](./data/saved/categories_25.txt) are visualized below using t-SNE (2 components, perplexity is 24). 

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./data/resources/word_selection_plot_dark.png">
        <img alt="" src="./data/resources/word_selection_plot_light.png" />
    </picture>
</p>


#### Data Collection
For each of the selected 25 words, I [prompt ChatGPT](./data/PromptManager.py) for a textual description of the given word. The prompt template used for ChatGPT is parameterized across six aspects. The template is as follows:

```diff
Give me [length] [detail] unique descriptions of [word]. Do not include the word [word] or any of its variations in your response. Use [complexity] language in your response. Start all your responses with [prefix].
```

The six parameters included in the template:
- Word: the given word that needs to be described by ChatGPT.
- Length: number of descriptions to generate per prompt for a given word. It will always generate 20 descriptions.
- Level of detail: length of each description for a given word. The possible values are: <em>very simple</em>, <em>simple</em>, <em>long</em>, <em>very long</em>, or blank (i.e., not specified).
- Complexity: type of diction used in each description for a given word (e.g., sophisticated vs. simple). The possible values are: <em>very simple</em>, <em>simple</em>, <em>complex</em>, <em>very complex</em>, or blank (i.e., not specified).
- Prefix: the first word that needs to be used in the description (designed to encourage different sentence structures). The possible values are: <em>it</em>, <em>this</em>, <em>a</em>, <em>the</em>, <em>with</em>, or blank (i.e., not specified). If the prefix is blank, then the last sentence in the prompt template is also removed.
- Temperature: value of the temperature variable used in call to ChatGPT's API. The possible values are: 0.2, 0.6, or 1.

I compute all possible combinations of these parameters for each word to generate [7200 descriptions per word](./data/saved/raw_descriptions.csv) (180,000 in total). 
The chosen sample size per word is based on MNIST (7000 images per digit).


#### Data Preparation
The [preprocessing pipeline](./data/PreprocessingPipeline.py) I use consists of the following sequence of steps:
1. Making all text lowercase.
2. Expanding all contractions (e.g., can't &rarr; can not).
3. Removing all stopwords (e.g., a, the, it, etc.).
4. Cleaning all text (e.g., punctuation, hyperlinks, etc.).

I also remove all duplicates and use label encoding. Lemmatization was initially explored as a method for text standardization, but it 
was ultimately discarded after experiments showed it reduced performance.

The [train-test-validation](./data/splits) split is 55%-30%-15%. It is a random split since there is a class balance, as shown below:

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="./data/resources/class_balance_chart_dark.png">
        <img alt="" src="./data/resources/class_balance_chart_light.png" />
    </picture>
</p>


### Model Development
I experiment with four different models: a [kNN](./models/kNN.py), [XGBoost](./models/XGBoost.py), [SVM](./models/SVM.py) and [LSTM](./models/LSTM.py).

#### Hyperparameter Tuning

Due to the size of the hyperparameter space of the LSTM and to enable fair comparison between models, 
I restrict the tuning process to a single train-validation split (as defined [here](#data-preparation)). 
I use grid search (kNN, XGBoost, SVM) and Bayesian optimization (LSTM).
The method used is determined by the number of hyperparameters, where grid search
is used for smaller search spaces and Bayesian optimization is used for larger ones.
For the LSTM model, I also perform some very light manual tuning afterward.


<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Type</th>
            <th>Hyperparameter</th>
            <th><span>Tuning</span>
method<span></span></th>
            <th>Range</th>
            <th>Auto-selected value</th>
            <th>Final value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2><a href="./results/knn/knn_tuning.json">kNN</a></td>
            <td colspan="2"># neighbours</td>
            <td rowspan="2">Grid search</td>
            <td>{1, 2, ..., 50}</td>
            <td colspan="2">35</td>
        </tr>
        <tr>
            <td colspan="2">Weight type</td>
            <td>{distance, uniform}</td>
            <td colspan="2">distance</td>
        </tr>
        <tr>
            <td rowspan=3><a href="./results/xgboost/xgboost_tuning.json">XGBoost</a> <sup>1</sup></td>
            <td colspan="2"># estimators</td>
            <td rowspan="3">Grid search</td>
            <td>{100, 1000}</td>
            <td colspan="2">1000</td>
        </tr>
        <tr>
            <td colspan="2">Learning rate</td>
            <td>{10<sup>-4</sup>, 10<sup>-3</sup>, ..., 10<sup>-1</sup>}</td>
            <td colspan="2">10<sup>-1</sup></td>
        </tr>
        <tr>
            <td colspan="2">Max depth</td>
            <td>{3, 5, ..., 13}</td>
            <td colspan="2">7</td>
        </tr>
        <tr>
            <td rowspan=2><a href="./results/svm/svm_tuning.json">SVM</a></td>
            <td colspan="2">$C$</td>
            <td rowspan="2">Grid search</td>
            <td>{10<sup>-1</sup>, 1, ..., 10<sup>3</sup>}</td>
            <td colspan="2">10</td>
        </tr>
        <tr>
            <td colspan="2">$\gamma$</td>
            <td>{10<sup>-4</sup>, 10<sup>-3</sup>, ..., 10<sup>1</sup>}</td>
            <td colspan="2">1</td>
        </tr>
        <tr>
            <td rowspan=10><a href="./results/lstm">LSTM</a> <sup>2</sup></td>
            <td rowspan=4><a href="./results/lstm/lstm_tuning_arch_1.json">Architecture</a></td>
            <td># LSTM units</td>
            <td rowspan="10">Bayesian optimization</td>
            <td>{64, 128, ..., 512}</td>
            <td colspan="2">448</td>
        </tr>
        <tr>
            <td># FC layers</td>
            <td>{1, 2}</td>
            <td colspan="2">1</td>
        </tr>
        <tr>
            <td># units per FC layer</td>
            <td>{128, 256, ..., 1024}</td>
            <td colspan="2">384</td>
        </tr>
        <tr>
            <td>Dropout per FC layer</td>
            <td>{0.1, 0.3, ..., 0.7}</td>
            <td>0.3</td>
            <td>0.7</td>
        </tr>
        <tr>
            <td rowspan=3><a href="./results/lstm/lstm_tuning_adam_1.json">LR Schedule</a></td>
            <td>Scheduler</td>
            <td colspan="3">Cosine Decay</td>
        </tr>
        <tr>
            <td>Initial LR</td>
            <td>[10<sup>-4</sup>, 10<sup>-3</sup>]</td>
            <td>9.99 $\times$ 10<sup>-4</sup></td>
            <td>10<sup>-3</sup></td>
        </tr>
        <tr>
            <td># decay steps</td>
            <td>{25, 50, ..., 250}</td>
            <td>250</td>
            <td>25</td>
        </tr>
        <tr>
            <td rowspan=3><a href="./results/lstm/lstm_tuning_adam_1.json">Optimizer</a></td>
            <td>Optimizer</td>
            <td>{Adam, SGD}</td>
            <td colspan="2">Adam</td>
        </tr>
        <tr>
            <td>$\beta_1$</td>
            <td>[0, 0.9999]</td>
            <td colspan="2">0.906</td>
        </tr>
        <tr>
            <td>$\beta_2$</td>
            <td>[0.9, 0.9999]</td>
            <td colspan="2">0.955</td>
        </tr>
    </tbody>
</table>

<sup>1</sup> The `tree_method` parameter is set to `hist` to ensure the model
can be used on a GPU.

<sup>2</sup> The loss function used for all LSTM experiments is sparse cross-entropy as the labels are not one-hot encoded.
Additionally, the batch size used is 256 trained across 10,000 epochs (early stopping patience of 10).

### Evaluation

#### Procedure

The procedure during model evaluation is as follows. I concatenate the training and validation data and perform
cross validation (5 splits) on this set. The best performing model (by accuracy) is selected which is then evaluated
on the test set. I mainly focus on accuracy as the dataset has a class balance, but the precision, recall and F1-score
are also reported for completeness. All model hyperparameters are identical to those used during tuning, except
for the LSTM model, which uses an early stopping patience of 20.

#### Results

The table below shows the results for all four models, with the best results highlighted in **bold**. The LSTM model performs best, followed by SVM, XGBoost and kNN.


<div align="center">
    <table>
        <thead>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-score</th>
        </thead>
        <tbody>
            <tr>
                <td><a href="./results/knn/knn_evaluation.json">kNN</a></td>
                <td>94.37</td>
                <td>94.42</td>
                <td>94.38</td>
                <td>94.38</td>
            </tr>
            <tr>
                <td><a href="./results/xgboost/xgboost_evaluation.json">XGBoost</a></td>
                <td>95.84</td>
                <td>95.86</td>
                <td>95.85</td>
                <td>95.85</td>
            </tr>
            <tr>
                <td><a href="./results/svm/svm_evaluation.json">SVM</a></td>
                <td>97.51</td>
                <td>97.53</td>
                <td>97.51</td>
                <td>97.52</td>
            </tr>
            <tr>
                <td><a href="./results/lstm/lstm_evaluation.json">LSTM</a></td>
                <td><b>97.75</b></td>
                <td><b>97.75</b></td>
                <td><b>97.75</b></td>
                <td><b>97.75</b></td>
            </tr>
        </tbody>
    </table>
</div>

The two graphs below show the loss and accuracy curves of the LSTM model. According to the loss, the model 
is overfitting, but the accuracy still keeps improving. Despite this, I still opt to use the LSTM
model in the [demo](https://descraibe-it.onrender.com/). This is because (1) the cost of missclassification 
in the context of the game is very high (resulting in game over), and (2) there is no need to generalize to 
new data since the game consists of a fixed number of words.

<picture>
    <source media="(prefers-color-scheme: dark)" srcset="./results/lstm/lstm_loss_acc_plot_dark.png">
    <img alt="" src="./results/lstm/lstm_loss_acc_plot_light.png" />
</picture>

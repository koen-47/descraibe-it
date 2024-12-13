<div align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="./data/resources/descraibe-it-header-dark.png">
    <img alt="" src="./data/resources/descraibe-it-header-light.png" />
</picture>
</div>

## Installation

### Reproducing the Results


## Methodology

### Data Overview
The dataset is created through three key phases: selection, collection, and preparation. Each phase is detailed below 
to provide a clear understanding of the data pipeline.

#### Data Selection
The first step towards collecting the dataset consists of selecting which words will 
need to be described by the player. We take inspiration from the game [Quick, Draw!](https://quickdraw.withgoogle.com/), 
a computer vision alternative which uses doodles instead of text descriptions. 
First, we select the [words](https://github.com/googlecreativelab/quickdraw-dataset/blob/master/categories.txt) used in that game.
As our approach to word selection is based on word embeddings, we remove all [entries](./data/saved/categories_289.txt) consisting of more than one word to ensure we can compare
the embeddings with each other.

From these remaining words, we further narrow them down to a subset of 25. This is done in the interest of 
the time and resource costs incurred by using OpenAI's API for large scale purposes, as many descriptions per word will be 
needed to accurately and fairly evaluate the validity of the more data hungry neural models. 

To select a suitable set of 25 words, we aim to create as semantically a diverse subset as possible.
This can be achieved by maximizing the minimum distance between the embeddings associated with each of the selected words.
More formally, for a set of words $W$, we want to find the subset of words $W' \subset W$ using the Euclidean
distance function $d(e(w_i), e(w_j))$, where $e(w)$ returns the embedding associated with word $w \in W$.
The subset $W'$ should satisfy the following:

$$
\mathop{\max} \left( \mathop{\min}\limits_{w_i, w_j \in W' \atop i \neq j} d \left( e(w_i), e(w_j) \right) \right)
$$

This problem is known as the [<em>max-min diversity problem</em>](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=51f97d822ec695f5700ac353bfff285bd44ef0e7) and is considered NP-hard. 
As such, we use a greedy algorithm that approximates the optimal subset. The embeddings we use are [pretrained GloVe word embeddings](https://nlp.stanford.edu/projects/glove/) (840B tokens + 300d vectors).

The [final set of selected words](./data/saved/categories_25.txt) are visualized below using t-SNE (2 components, perplexity is 24). 

![word_selection_plot](./data/resources/word_selection_plot_dark.png)


#### Data Collection
For each of the selected 25 words, we [prompt ChatGPT](./data/PromptManager.py) for a textual description of the given word. The prompt template used for ChatGPT is parameterized across six aspects. The template is as follows:

```diff
Give me [length] [detail] unique descriptions of [word]. Do not include the word [word] or any of its variations in your response. Use [complexity] language in your response. Start all your responses with [prefix].
```

The six parameters included in the template:
- Word: the given word that needs to be described by ChatGPT.
- Length: number of descriptions to generate per prompt for a given word. It will always generate 20 descriptions.
- Level of detail: length of each description for a given word. The possible values are: <em>very simple</em>, <em>simple</em>, <em>long</em>, <em>very long</em>, or blank (i.e., not specified).
- Complexity: type of diction used in each description for a given word (e.g., sophisticated vs. simple). The possible values are: <em>very simple</em>, <em>simple</em>, <em>complex</em>, <em>very complex</em>, or blank (i.e., not specified).
- Prefix: the first word that needs to be used in the description (designed to encourage difference sentence structures). The possible values are: <em>it</em>, <em>this</em>, <em>a</em>, <em>the</em>, <em>with</em>, or blank (i.e., not specified). If the prefix is blank, then the last sentence in the prompt template is also removed.
- Temperature: value of the temperature variable used in call to ChatGPT's API. The possible values are: 0.2, 0.6, or 1.

We compute all possible combinations of these parameters for each word to generate [7200 descriptions per word](./data/saved/raw_descriptions.csv) (180,000 in total). 
The chosen sample size per word is based on MNIST (7000 images per digit).


#### Data Preparation
The [preprocessing pipeline](./data/PreprocessingPipeline.py) we use consists of the following sequence of steps:
1. Making all text lowercase.
2. Expanding all contractions (e.g., can't &rarr; can not).
3. Removing all stopwords (e.g., a, the, it, etc.).
4. Cleaning all text (e.g., punctuation, hyperlinks, etc.).

We also remove all duplicates and use label encoding. Lemmatization was initially explored as a method for text standardization, but it 
was ultimately discarded after experiments showed it reduced performance.

The [train-test-validation](./data/splits) split is 55%-30%-15%. It is a random split since there is a class balance, as shown below:

<p align="center">
  <img src="./data/resources/class_balance_chart_dark.png" />
</p>


### Model Development
We experiment with three different models: a kNN, SVM and LSTM.

#### Hyperparameter Tuning

We use Grid Search (kNN, SVM) and Bayesian Optimization (LSTM) to perform hyperparameter tuning.
The method used is determined by the number of hyperparameters, where Grid Search
is used for smaller search spaces and Bayesian Optimization is used for larger ones.


<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Hyperparameter Type</th>
            <th>Hyperparameter</th>
            <th>Tuning Method</th>
            <th>Range</th>
            <th>Selected value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>kNN</td>
            <td colspan="2"># neighbours</td>
            <td rowspan="2">Grid Search</td>
            <td>{1, 2, ..., 50}</td>
            <td>35</td>
        </tr>
        <tr>
            <td colspan="2">Weight type</td>
            <td>{distance, uniform}</td>
            <td>distance</td>
        </tr>
        <tr>
            <td rowspan=2>SVM</td>
            <td colspan="2">C</td>
            <td rowspan="2">Grid Search</td>
            <td>{0.1, 1, ..., 1000}</td>
            <td>10</td>
        </tr>
        <tr>
            <td colspan="2">γ</td>
            <td>{0.0001, 0.001, ..., 10}</td>
            <td>1</td>
        </tr>
        <tr>
            <td rowspan=9>LSTM</td>
            <td rowspan=3>Architecture</td>
            <td># LSTM units</td>
            <td rowspan="9">Bayesian Optimization</td>
        </tr>
        <tr>
            <td># FC layers</td>
            <td>-</td>
        </tr>
        <tr>
            <td># units per FC layer</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan=3>LR Schedule</td>
            <td>Scheduler</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Initial LR</td>
            <td>-</td>
        </tr>
        <tr>
            <td># decay steps</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan=3>Optimizer</td>
            <td>Optimizer</td>
            <td>-</td>
        </tr>
        <tr>
            <td>β ₁</td>
            <td>-</td>
        </tr>
        <tr>
            <td>β ₂</td>
            <td>-</td>
        </tr>
    </tbody>
</table>

### Results
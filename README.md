<div align="center">
<h1>descr<span style="color: rgb(255, 71, 71)">AI</span>be it</h1>
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
From these remaining words, we aim to take a subset of words that are most 
different to each other. That is, we aim to maximize the minimum distance 
between the embeddings associated with each word. More formally, 
for the set of embeddings $E$, we want to find the subset of embeddings $E' \subset E$
that adhere to the following equation:

$$
\hat{\theta}^{MLE}=\underset{a}{\operatorname{\argmax}} P(D|\theta) = \frac{a_1}{a_1+a_0}
$$

![asdf](./data/resources/descraibe-it_word_selection.png)

### Model Development

### Results
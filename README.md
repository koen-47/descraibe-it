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


From these remaining words, we aim to find the subset of words that maximizes the 
semantic differences between them. This is performed by finding the subset of words that
maximizes the minimum distance of the embeddings associated with each word.

More formally, 
for a set of words $W$, we want to find the subset of words $W' \subset W$ using the Euclidean
distance function $d(e(i), e(j))$, where $e(w)$ returns the embedding associated with word $w$.
The subset $W'$ should satisfy the following equation:

$$
\mathop{\max} \mathop{\min}\limits_{i \neq j} d(e(i), e(i))
$$

![asdf](./data/resources/descraibe-it_word_selection.png)

### Model Development

### Results
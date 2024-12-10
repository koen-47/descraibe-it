import numpy as np
from tqdm import tqdm


class GloVeEmbedding:
    """
    Class that handles all functionality related to GloVe embeddings.
    """

    def __init__(self, file_path, dimensionality):
        """
        Constructor for the GloVe embedding class.
        :param file_path: File path to where the GloVe embeddings are stored.
        """
        if file_path is not None:
            self.embedding_index = self.__get_embedding_index(file_path)
        self.dimensionality = dimensionality

    def __get_embedding_index(self, file_path):
        """
        Parses the file containing the GloVe embeddings and returns its word index.
        :param file_path: File path to where the GloVe embeddings are stored.
        :return: Dictionary containing the word index of the GloVe embeddings file.
        """
        embeddings_index = dict()
        with open(file_path, encoding="utf-8") as file:
            num_lines = sum(1 for _ in file)

        file = open(file_path, encoding="utf8")
        for line in tqdm(file, total=num_lines, desc="Loading GloVe embedding"):
            values = line.strip().split(" ")
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        file.close()
        return embeddings_index

    def compute_embedding_matrix(self, tokenizer, size_of_vocabulary):
        """
        Computes the embedding matrix to be used in the embedding layer of a neural model.
        :param tokenizer: Keras Tokenizer that has been fit to the training data.
        :param size_of_vocabulary: Vocabulary size (integer)
        :return: Returns the embedding matrix.
        """
        embedding_matrix = np.zeros((size_of_vocabulary, self.dimensionality))

        for word, i in tokenizer.word_index.items():
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def __getitem__(self, item):
        return self.embedding_index[item]


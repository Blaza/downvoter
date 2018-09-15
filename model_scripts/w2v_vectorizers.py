# original source code:
# http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/

from nltk import word_tokenize
import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.wv.vector_size

    def fit(self, X, y):
        return self

    def transform(self, X):
        """
        Transform the texts given in X by taking vector representations of each
        word in the text and averaging them.
        """
        return np.array([
            np.mean([self.word2vec[w] for w in word_tokenize(words)
                     if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# coding: utf-8
"""
 TODO: add word vectors initialization with GloVE/word2vec/whatever
"""
import numpy as np


def get_vectors(word2id={}, vectors_path="glove.6B.50d.txt"):

    encoder_input_data = np.random.random((len(word2id), 50))

    for line in open(vectors_path):
        line = line.strip().split(" ")
        if line[0] in word2id:
            encoder_input_data[word2id[line[0]]] = np.array([float(n) for n in line[1:]])

    return encoder_input_data

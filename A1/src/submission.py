#!/usr/bin/env python

import sys
import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join('..')))

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    # ### START CODE HERE ###
    out = np.concatenate(corpus).ravel()
    corpus_words = sorted(np.unique(out))
    num_corpus_words = len(corpus_words)
    # ### END CODE HERE ###
    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurrence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """

    # ### START CODE HERE ###

    # 1. iterate lists in corpus
    # 2. add padding to lists, o.w. indexing won't work
    # 3. iterate lists and find center words and words to left and right of center word
    # 4. find indices for cooccuring words and add 1 to correct index

    words, num_words = distinct_words(corpus)
    M = np.zeros((num_words, num_words))
    word2Ind = {word:i for i,word in enumerate(words)}

    for word_ls in corpus:
        word_ls = [None for i in range(window_size)] + word_ls + [None for i in range(window_size)]  # padding
        for i in range(window_size, len(word_ls) - window_size):

            word_slice = word_ls[i - window_size:i + window_size + 1]
            cut = len(word_slice) // 2

            center_word = word_slice[window_size]
            word_slice_left = word_slice[:cut]
            word_slice_right = word_slice[cut + 1:]

            # print("For word slice:", word_slice)
            # print("left:",word_slice_left)
            # print("center:",center_word)
            # print("right:",word_slice_right)

            # list of N indices for word M in MxN matrix
            words_to_add = [word2Ind[tok] for tok in word_slice_left + word_slice_right
                            if tok is not None]
            # print("Words to add:", words_to_add)
            # print("--------------------------")

            # index of word M
            m_idx = word2Ind[center_word]

            for word_idx in words_to_add:
                M[m_idx, word_idx] += 1
    # ### END CODE HERE ###

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of corpus words)): co-occurrence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    np.random.seed(4355)
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ### START CODE HERE ###
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.fit_transform(M)
    # ### END CODE HERE ###

    print("Done.")
    return M_reduced

def main():
    matplotlib.use('agg')
    plt.rcParams['figure.figsize'] = [10, 5]

    assert sys.version_info[0] == 3
    assert sys.version_info[1] >= 5

    def plot_embeddings(M_reduced, word2Ind, words, title):

        for word in words:
            idx = word2Ind[word]
            x = M_reduced[idx, 0]
            y = M_reduced[idx, 1]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, word, fontsize=9)
        plt.savefig(title)

    #Read in the corpus
    reuters_corpus = read_corpus()

    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, 'co_occurrence_embeddings_(soln).png')

if __name__ == "__main__":
    main()

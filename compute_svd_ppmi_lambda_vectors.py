#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in February 2020

@author: Jakob Jungmaier

Calculates word embeddings from corpus by using PPMI, SVD,
and Dirichlet Smoothing. For more details cf. Jungmaier/Kassner/Roth(2020):
"Dirichlet-Smoothed Word Embeddings for Low-Resource Settings"
"""

import argparse
import math
import numpy as np
import random
from collections import defaultdict

from scipy.sparse import csr_matrix, dok_matrix
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def file_to_cooc_matrix(file_name, chunk_size=3000000, window_size=5,
                        min_count=1, subsampling_rate=0.00001, verbose=True):
    """
    Takes a text corpus file (with one word per line (!)) and returns
    a scipy sparse co-occurrence matrix.
    Processes the text file in chunks to save memory -> better for large
    corpus files.

    Parameters:

    chunk_size: approximate size in bytes of the chunks to be processed from
                the corpus (does not cut lines),
                smaller chunks -> less memory usage but slower
    window_size: window size of the co-occurrence window around a word ->
                 maximum distance to middle word is window_size (default: 5)
    min_count: minimum count of words in the corpus to consider
               (default 1 -> all words are considered)
    subsampling_rate: subsample more frequent words (similar to word2vec)
                      (default: 0.00001)
    verbose: verbose output
    """
    word_count = defaultdict(int)

    # Counting chunks to process
    if verbose:
        print("Counting chunks, computing vocabulary...")

    with open(file_name) as corpus_file:
        chunks_total = 0
        while True:
            text_chunk = corpus_file.readlines(chunk_size)
            if not text_chunk:
                break
            chunks_total += 1
            for word in text_chunk:
                word_count[word.rstrip()] += 1

        # Computing vocabulary of words appearing at least min_count
        # times (ordered according to descending frequency)
        vocab = [word for word, count in sorted(word_count.items(),
                                                key=lambda x:x[1],
                                                reverse=True)
                 if count >= min_count]
        vocab_set = set(vocab)

        # Preparing subsampling...
        if subsampling_rate:
            corpus_size = sum(word_count.values())
            subsampling_rate = subsampling_rate * corpus_size
            subsampling_dict = {word: 1-math.sqrt(subsampling_rate/count)
                                for word, count in word_count.items()
                                if count > subsampling_rate}
            rand = random.Random(0)
            if verbose:
                print("Corpus size: {}".format(corpus_size))
                print("Subsampled words: {}".format(subsampling_rate))
                print("Words in subsampling \
                       dictionary: {}".format(len(subsampling_dict)))

        if verbose:
            print("Chunks to process: {}".format(chunks_total))
            print("Vocabulary: {}".format(len(vocab)))
            print("Computing co-occurrence matrix:")

        # Reading file from beginning again
        corpus_file.seek(0, 0)

        # Generating word->id mapping
        word_to_id = {w: i for i, w in enumerate(vocab)}

        # Initializing co-occurrence matrix
        m = csr_matrix((len(vocab), len(vocab)), dtype=int)

        # Reading and processing text chunkwise for saving memory
        chunk_count = 0
        while True:
            text_chunk = corpus_file.readlines(chunk_size)
            if not text_chunk:
                break
            chunk_count += 1

            if verbose:
                if chunk_count != chunks_total:
                    print("Processing chunk {} of {}".format(chunk_count,
                                                             chunks_total),
                          end="\r")
                else:
                    print("Processing chunk {} of {}".format(chunk_count,
                                                             chunks_total))

            # Ignoring words not in vocabulary (similar to word2vec)
            text_chunk = [word.rstrip() for word in text_chunk
                          if word.rstrip() in vocab_set]

            # Subsampling
            if subsampling_rate:
                text_chunk = [word for word in text_chunk
                              if (word not in subsampling_dict or
                                  rand.random() > subsampling_dict[word])]

            # Collecting co-occurrences of the present chunk
            row = []
            col = []
            data = []

            # Going through each word in chunk -> middle word
            for middle_position in range(len(text_chunk)):
                middle_word = text_chunk[middle_position]
                context_start = max(0, middle_position - window_size)
                context_end = min(len(text_chunk), middle_position
                                  + window_size + 1)
                # Going through each word in the context window_size
                # around the middle word
                for context_position in range(context_start, context_end):
                    context_word = text_chunk[context_position]
                    # Doesn't take the same position into account,
                    # but a word might still co-occur with itself and
                    # yield a pair (w1,w1) -> slightly better results
                    if context_position != middle_position:
                        context_word = text_chunk[context_position]
                        row.append(word_to_id[middle_word])
                        col.append(word_to_id[context_word])
                        data.append(1)

            # Storing co-occurrence counts in a temporary matrix for
            # the present chunk
            tmp_m = csr_matrix((data, (row, col)),
                               shape=(len(vocab), len(vocab)), dtype=float)

            # Updating co-occurrence matrix with entries of temporary matrix
            m = m + tmp_m

        if verbose:
            print("Matrix size: {}".format(np.prod(m.shape)))
            print("Non-zero elements: {}".format(m.nnz))

    return m, word_to_id


def save_word_vectors(file_name, word_vector_matrix, word_to_id, vocab,
                      verbose=True):
    """
    Saves word vectors from a word vector matrix to a text file (in word2vec
    format):

    #(vectors) #(dimensions)
    word1 dim1 dim2 dim3 ...
    word2 dim1 dim2 dim3 ...
    .     .    .    .
    .     .    .    .
    .     .    .    .

    Parameters:

    file_name: name of the word vector file
    word_vector_matrix: matrix containing word vectors, rows correspond to
                        words
    word_to_id: mapping from words to ids (dictionary)
    vocab: vocabulary for which to save the word vectors,
            vectors will be saved in the order of the vocabulary list
    verbose: verbose output
    """
    if verbose:
        print("Saving word vectors for {} most frequent words:"
              .format(len(vocab)))

    with open(file_name, "w") as vector_file:
        vector_file.write(str(word_vector_matrix.shape[0]) + " "
                          + str(word_vector_matrix.shape[1]) + "\n")

        for i, word in enumerate(vocab, start=1):
            vector_file.write(word
                              + " "
                              + " ".join([str(value)
                                          for value
                                          in word_vector_matrix
                                          [word_to_id[word], :]])
                              + "\n")
            if verbose:
                if i % 1000 == 0:
                    print("{} of {} word vectors saved.".format(i,
                                                                len(vocab)),
                          end="\r")
                elif i == len(vocab):
                    print("{} of {} word vectors saved.".format(i, len(vocab)))


def pmi_weight(cooc_matrix, smoothing_factor=0, threshold=0, verbose=True):
    """
    Calculates (P)PMI matrix with Dirichlet smoothing

    Parameters:

    cooc_matrix: scipy sparse co-occurrence matrix
    amoothing_factor: smoothing factor lambda (default: 0 -> no smoothing)
    threshold: threshold for cutting off values (default: 0 -> PPMI)
    verbose: verbose output

    Adapted in parts from Omer Levy: https://bitbucket.org/omerlevy/hyperwords/
                                     (accessed on 30/05/19)
    """
    if verbose:
        print("PMI weighting:")
    if smoothing_factor != 0:
        if verbose:
            print("Smoothing with lambda={}".format(smoothing_factor))
        sum_w = np.array(cooc_matrix.sum(axis=1))[:, 0] \
                + (smoothing_factor * cooc_matrix.shape[0])
        sum_c = np.array(cooc_matrix.sum(axis=0))[0, :] \
                + (smoothing_factor * cooc_matrix.shape[0])
    else:
        if verbose:
            print("No smoothing.")
        sum_w = np.array(cooc_matrix.sum(axis=1))[:, 0]
        sum_c = np.array(cooc_matrix.sum(axis=0))[0, :]

    sum_total = sum_c.sum()

    if smoothing_factor != 0:
        cooc_matrix.data = cooc_matrix.data + smoothing_factor

    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    pmi = csr_matrix(cooc_matrix)
    pmi = multiply_by_rows(pmi, sum_w)
    pmi = multiply_by_columns(pmi, sum_c)
    pmi *= sum_total
    pmi.data = np.log(pmi.data)

    # setting the threshold:
    if threshold is not None:
        if verbose:
            print("Filtering values below {}".format(threshold))
        pmi.data[pmi.data < threshold] = 0

    return pmi


def multiply_by_rows(matrix, row_coefs):
    """
    From Omer Levy: https://bitbucket.org/omerlevy/hyperwords/
                    (accessed on 30/05/19)
    """
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    """
    From Omer Levy: https://bitbucket.org/omerlevy/hyperwords/
                    (accessed on 30/05/19)
    """
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Calculates word embeddings \
                                                  from corpus by using PPMI, \
                                                  SVD, and Dirichlet \
                                                  Smoothing. Cf.\
                                                  Jungmaier/Kassner/Roth \
                                                  (2020): "Dirichlet-Smoothed \
                                                  Word Embeddings for \
                                                  Low-Resource Settings"')
    parser.add_argument('corpus_file',
                        help='Text file with word tokens separated \
                              by newlines.')
    parser.add_argument('word_vector_filename',
                        help='Desired name of the word vector file.')
    parser.add_argument('--window_size', '-w', type=int, default=5,
                        help='Size of the window around a word for \
                              co-occurrence counts, integer. Default: 5.')
    parser.add_argument('--min_count', '-m', type=int, default=1,
                        help='Minimal word count for words to process, \
                              integer, useful range 1-10. Default: 1.')
    parser.add_argument('--subsampling', '-s', type=float, default=0.0,
                        help='Subsampling rate (similar to word2vec). \
                              Default: 0.0 (no subsampling).')
    parser.add_argument('--chunk_size', '-c', type=int, default=3000000,
                        help='Chunk size in bytes for chunkwise processing \
                              of the corpus. Default: 3000000. If memory \
                              overflow encountered, choose lower value.')
    parser.add_argument('--dimensions', '-d', type=int, default=100,
                        help='Size of word embeddings, integer. Default: 100.')
    parser.add_argument('--smoothing_factor', '-l', type=float, default=0.0001,
                        help='Smoothing factor lambda, float, \
                              useful range: 0-1. Default: 0.0001.')
    parser.add_argument('--threshold', '-t', type=float, default=0.0,
                        help='Threshold for PMI values to be cut off:\
                              Default: 0 -> PPMI.')
    parser.add_argument('--eigenvalue_weighting', '-e', type=float,
                        default=0.0, help='Weighting of singular values \
                                           (Sigma), float, useful range: 0-1. \
                                           Default: 0.0 (singular values are \
                                           not considered).')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output.')
    args = parser.parse_args()


    print(args)

    # Creating co-occurrence matrix and word-to-id mapping from the corpus
    m, word_to_id = file_to_cooc_matrix(args.corpus_file,
                                        chunk_size=args.chunk_size,
                                        window_size=args.window_size,
                                        min_count=args.min_count,
                                        subsampling_rate=args.subsampling,
                                        verbose=args.verbose)
    vocab = list(word_to_id.keys())

    # (P)PMI weighting
    m = pmi_weight(m, smoothing_factor=args.smoothing_factor,
                   threshold=args.threshold, verbose=args.verbose)

    # Singular value decomposition of the co-occurrence matrix
    if args.verbose:
        print("SVD...", end="\r")
    if args.eigenvalue_weighting == 1:
        svd = TruncatedSVD(n_components=args.dimensions, random_state=0)
        m = svd.fit_transform(m)
    elif args.eigenvalue_weighting == 0:
        m, _, _ = randomized_svd(m, n_components=args.dimensions,
                                 random_state=0)
    else:
        m, s, _ = randomized_svd(m, n_components=args.dimensions,
                                 random_state=0)
        sigma = np.zeros((m.shape[0], m.shape[1]))
        sigma = np.diag(s**args.eigenvalue_weighting)
        m = m.dot(sigma)
    if args.verbose:
        print("SVD...done.")

    # Normalizing word vectors
    if args.verbose:
        print("Normalizing vectors...", end="\r")
    m = normalize(m, norm='l2', axis=1, copy=False)
    if args.verbose:
        print("Normalizing vectors...done.")

    # Saving word vectors
    save_word_vectors(args.word_vector_filename, m, word_to_id, vocab,
                      verbose=args.verbose)

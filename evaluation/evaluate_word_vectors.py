#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in February 2020

@author: Jakob Jungmaier

Evaluates word vectors on the word similarity task by Pearson\'s and
Spearman\'s rank correlation coefficients.
"""

import argparse
from gensim.models import KeyedVectors


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluates word vectors on \
                                                  the word similarity task by \
                                                  Pearson\'s and Spearman\'s \
                                                  rank correlation \
                                                  coefficients.')
    parser.add_argument('word_vector_file', type=str, help='Word vector file \
                                                            in word2vec tex t\
                                                            format.')
    parser.add_argument('--datasets', '-d', nargs='+', help='Word similarity \
                                                             datasets.\
                                                             Format: Three \
                                                             tab separated \
                                                             columns "word1 \
                                                             \\t word2 \
                                                             \\t similarity \
                                                             score". At least \
                                                             one dataset is \
                                                             required.',
                        required=True)
    parser.add_argument('--dummy4unknown', '-d4u', action='store_true',
                        help='Use dummy4unknown, i.e., word pairs for which \
                              at least one corresponding vector is missing \
                              are evaluated as having zero accuracy. If not \
                              used, pairs will not be considered in \
                              evaluation. Default: False.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output.')
    args = parser.parse_args()

    if args.verbose:
        print("Reading word vectors from file...", end="\r")
    word_vectors = KeyedVectors.load_word2vec_format(args.word_vector_file,
                                                     binary=False)
    if args.verbose:
        print("Reading word vectors from file...done")

    datasets = args.datasets
    spearman_scores = []
    pearson_scores = []

    for dataset in datasets:
        results = word_vectors.evaluate_word_pairs(dataset,
                                                   restrict_vocab=10000000,
                                                   dummy4unknown
                                                   =args.dummy4unknown)
        pearson, spearman, oov_ratio = results

        spearman_scores.append(spearman[0])
        pearson_scores.append(pearson[0])

        print("\nDataset: {}".format(dataset.split('/')[-1].split('.')[0]))
        print("Spearman coefficient: {}".format(round(spearman[0], 3)))
        print("Pearson coefficient: {}".format(round(pearson[0], 3)))
        print("OOV ratio: {}".format(round(oov_ratio, 3), "%"))

    print("\nAverage Spearman coefficient: {}".format(
         round(sum(spearman_scores) / len(spearman_scores), 3)))
    print("Average Pearson coefficient: {}".format(round(sum(pearson_scores)
                                                         / len(pearson_scores),
                                                         3)))

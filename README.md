# Dirichlet-Smoothed Word Embeddings (for Low-Resource Settings)

Calculating word embeddings from a corpus by SVD-factorized co-occurrence matrices with dirichlet-smoothed PPMI weighting.

For details, please refer to [this paper](https://www.aclweb.org/anthology/2020.lrec-1.437.pdf).

## Abstract

Nowadays, classical count-based word embeddings using positive pointwise mutual information (PPMI) weighted co occurrence matrices have been widely superseded by machine-learning-based methods like word2vec and GloVe. But these methods are usually applied using very large amounts of text data. In many cases, however, there is not much text data available, for example for specificdomains or low-resource languages. This paper revisits PPMI by adding Dirichlet smoothing to correct its bias towards rare words. We evaluate on standard word similarity data sets and compare to word2vec and the recent state of the art for low-resource settings: Positive and Unlabeled (PU) Learning for word embeddings. The proposed method outperforms PU-Learning for low-resource settings and obtains competitive results for Maltese and Luxembourgish.

## Prerequisites

* Python 3.6 or newer
* Python module Scikit-learn
* Python module Gensim (for evaluation)

## Demo

To reproduce the paper's results on the full enwik9 corpus (-> third line in tab. 2) , please run

```
$ bash ./demo.sh
```

This will 

1. download and preprocess the corpus
2. compute dirichlet-smoothed word embeddings from it, and
3. evaluate the resulting word embeddings on the word similarity task.

Note that this will need around 2.4 GB of disk space for the corpus and the word embedding file.

## Usage

To compute word embeddings from any corpus, run

```
$ python3 compute_svd_ppmi_lambda_vectors.py corpus_file word_vector_filename
```

The input corpus_file has to be a file with one word per line.

For an overview of hyperparameters, call

```
$ python3 compute_svd_ppmi_lambda_vectors.py -h
```

For an evaluation of word embeddings (in word2vec text format) on the word similarity task, switch to the evaluation directory and run

```
$ python3 evaluate_word_vectors.py word_vector_file -d dataset1 dataset2 ...
```

where dataset1, dataset2, ... are word similarity datasets with three tab separated columns "word1 word2 similarity".

## Word Similarity Datasets

Usual word similarity datasets for English can be found in /evaluation/datasets.

Datasets for the evaluation of Luxembourgish and Maltese word embeddings (as used in the paper) which were translated from English by the Google Translation API, can be found in /evaluation/translated_datasets.

## Reference

Pleace cite:

```
@inproceedings{jungmaier-etal-2020-dirichlet,
    title = "{D}irichlet-Smoothed Word Embeddings for Low-Resource Settings",
    author = "Jungmaier, Jakob  and
      Kassner, Nora  and
      Roth, Benjamin",
    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.lrec-1.437",
    pages = "3560--3565",
}
```

## Acknowledgments

Thanks to Matt Mahoney for providing the enwik9 corpus and the Perl script "wikifil.pl" for preprocessing used in the demo script and to Omer Levy for parts of the code for computing the PPMI weights.

## License

This project is licensed under the terms of the GNU General Public License 3.

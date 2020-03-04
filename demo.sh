#!/bin/bash
# 
# Calculates and evaluates dirichlet-smoothed word embeddings for the enwik9
# corpus
#
# Usage: bash ./demo.sh
# -----------------------------------------------------------------------------

printf "Please make sure to use Python 3.6 or newer, and to have Python
module Gensim installed.\n\n"

if [ ! -e enwik9 ]; then
  printf "Downloading enwik9 corpus:\n\n"
  if hash wget 2>/dev/null; then
    wget http://mattmahoney.net/dc/enwik9.zip
  else
    curl -O http://mattmahoney.net/dc/enwik9.zip
  fi
  
  unzip enwik9.zip
  rm enwik9.zip
  
  printf "Preprocessing enwik9 corpus...\n\n"
  perl utils/wikifil.pl enwik9 > enwik9_clean
  rm enwik9
  cat enwik9_clean | tr " " "\n" | tail +2 > enwik9
  rm enwik9_clean
else
  printf "Corpus found.\n\n"
fi

# Hyperparameters for word embedding
WINDOW_SIZE=5
MIN_COUNT=1
SUBSAMPLING=0
CHUNK_SIZE=3000000
DIMENSIONS=100
SMOOTHING_FACTOR=0.01  # best value for last 60M words of enwik9
THRESHOLD=0
EIGENVALUE_WEIGHTING=0
VERBOSE="true"

# Hyperparameters for evaluation
DUMMY4UNKNOWN="true"
DATASETS='evaluation/datasets/RG-65.tab evaluation/datasets/WordSim-353.tab
evaluation/datasets/SimLex-999.tab evaluation/datasets/MEN.tab
evaluation/datasets/RW.tab'

printf "Computing word embeddings from corpus:\n\n"
if [ $VERBOSE ]; then
  python3 compute_svd_ppmi_lambda_vectors.py enwik9 enwik9_demo_vectors \
          -w $WINDOW_SIZE -m $MIN_COUNT -s $SUBSAMPLING \
          -c $CHUNK_SIZE -d $DIMENSIONS -l $SMOOTHING_FACTOR -t $THRESHOLD \
          -e $EIGENVALUE_WEIGHTING -v
  printf "\nEvaluating word vectors:\n\n"
  if [ $DUMMY4UNKNOWN ]; then
    python3 evaluation/evaluate_word_vectors.py enwik9_demo_vectors \
            -d $DATASETS -v -d4u
  else
    python3 evaluation/evaluate_word_vectors.py enwik9_demo_vectors \
            -d $DATASETS -v
  fi
else
  python3 compute_svd_ppmi_lambda_vectors.py enwik9 enwik9_demo_vectors \
          -w $WINDOW_SIZE -m $MIN_COUNT -s $SUBSAMPLING \
          -c $CHUNK_SIZE -d $DIMENSIONS -l $SMOOTHING_FACTOR -t $THRESHOLD \
          -e $EIGENVALUE_WEIGHTING
  printf "\nEvaluating word vectors:\n\n"
  if [ $DUMMY4UNKNOWN ]; then
    python3 evaluation/evaluate_word_vectors.py enwik9_demo_vectors \
            -d $DATASETS -d4u
  else
    python3 evaluation/evaluate_word_vectors.py enwik9_demo_vectors \
            -d $DATASETS
  fi
fi

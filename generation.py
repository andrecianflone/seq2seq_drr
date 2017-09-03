"""
-----------
Description
-----------
Implicit DRR with Seq2Seq with Attention
Author: Andre Cianflone

For a single trial, call script without arguments

-----------
"""
from helper import Preprocess, Data, MiniData, make_batches, settings, alignment
from enc_dec import EncDecGen, EncDecClass
from training import train
from embeddings import Embeddings
import tensorflow as tf
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from enc_dec import EncDec
import sys
from six.moves import cPickle as pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from pprint import pprint
import codecs
import json
import argparse

###############################################################################
# Data
###############################################################################

global hparams
hparams, settings = settings('settings.json')

# Settings file
dataset_name = settings['use_dataset']

data_class = Preprocess(
            # dataset_name='conll',
            dataset_name = dataset_name,
            relation = settings[dataset_name]['this_relation'],
            max_vocab = settings['max_vocab'],
            random_negative=False,
            max_arg_len= hparams.max_arg_len,
            maxlen=hparams.maxlen,
            settings=settings,
            split_input=True,
            pad_tag = hparams.pad_tag,
            unknown_tag = hparams.unknown_tag,
            bos_tag = hparams.bos_tag,
            eos_tag = hparams.eos_tag)

vocab = data_class.vocab
inv_vocab = data_class.inv_vocab

# Get the integer value for bos and eos tags
hparams.update(
  start_token = vocab[hparams.bos_tag],
  end_token = vocab[hparams.eos_tag],
)

# Data sets as Data objects
dataset_dict = data_class.data_collect

# Word embeddings
emb = Embeddings(
        vocab,
        inv_vocab,
        random_init_unknown=settings['random_init_unknown'],
        unknown_tag = hparams.unknown_tag)

# embedding is a numpy array [vocab size x embedding dimension]
embedding = emb.get_embedding_matrix(\
            word2vec_model_path=settings['embedding']['model_path'],
            small_model_path=settings['embedding']['small_model_path'],
            save=True,
            load_saved=True)

###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__,
                          formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('--task', default="generation", help='generation or classification')
  args = parser.parse_args()

  if args.task == 'generation':
    model = EncDecGen
  else:
    model = EncDecClass

  train(hparams, model, embedding, 300, dataset_dict, vocab, inv_vocab)




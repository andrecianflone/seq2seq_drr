"""
-----------
Description
-----------
Implicit DRR with Seq2Seq with Attention
Author: Andre Cianflone

For a single trial, call script without arguments

-----------
"""
from helper import settings, get_data
from embeddings import get_embeddings
from enc_dec import EncDecGen, EncDecClass
from training import train
import argparse

###############################################################################
# Data
###############################################################################
global hparams
hparams, settings = settings('settings.json')

# Get data
# dataset dictionary {k: v} is {dataset name: Data object}
dataset_dict, vocab, inv_vocab = get_data(hparams, settings)
# Embedding as numpy array, and embedding size
embedding, emb_dim = get_embeddings(hparams, vocab, inv_vocab, settings)

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

  train(hparams, settings, model, embedding, emb_dim, dataset_dict, vocab, inv_vocab)


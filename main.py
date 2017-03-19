"""
Author: Andre Cianflone
Encoder Decoder type model for DRR
"""
from helper import Data
from embeddings import Embeddings
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle as group_shuffle
from enc_dec import BasicEncDec
sk_seed = 0

# Some hyperparams
nb_epoch       = 2               # max training epochs
batch_size     = 32              # training batch size
max_arg_len    = 60              # max length of each arg
maxlen         = max_arg_len * 2 # max num of tokens per sample
num_units      = 4               # hidden layer size
num_layers     = 2               # try bidir?
max_time_steps = 100

###############################################################################
# Data
###############################################################################
conll_data = Data(max_arg_len=max_arg_len, maxlen=maxlen)
# X is a list of narrays: [arg1, arg2] , args are integers
# y is a list of narrays: [label_low_level, label_1st_level]
(X_train, y_train), (X_test, y_test) = conll_data.get_data()

# Sequence length, [length_arg1, legnth_arg2]
seq_len_train, seq_len_val = conll_data.get_seq_length()

# For mask to work, padding must be integer 0
train_dec_mask = np.sign(X_train[1])
test_dec_mask  = np.sign(X_test[1])

emb = Embeddings(conll_data.vocab, conll_data.inv_vocab)
# embedding is a numpy array [vocab size x embedding dimension]
embedding = emb.get_embedding_matrix(\
            model_path='data/google_news_300.bin',
            save=True,
            load_saved=True)


###############################################################################
# Main stuff
###############################################################################
model = BasicEncDec(\
        num_units=num_units,
        max_seq_len=max_arg_len,
        embedding=embedding)

def make_batches(data, batch_size, shuffle=True):
  """ Batches the passed data, even if data is a list of numpy arrays
  Args:
    data       : a list of numpy arrays
    batch_size : int
    shuffle    : should be true except when testing
  Returns:
    list of original numpy arrays but sliced
  """
  sk_seed = np.random.randint(0,10000)
  if shuffle: data = group_shuffle(data, random_state=sk_seed)
  data_size = len(data)
  batch_per_epoch = int(data_size/batch_size) + 1
  for epoch in range(num_epochs):
    for batch_num in range(batch_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield data[start_index:end_index]

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for epoch in nb_epochs:
    data = [X_train, y_train, seq_len_train]
    batches = make_batches(data, batch_size,shuffle=True)
    for batch in batches:
      fetch = [model.optimizer, model.cost]
      # feed = {enc_input: , enc_input_len: , dec_input:, dec_input_len:, dec_input_weight_mask}

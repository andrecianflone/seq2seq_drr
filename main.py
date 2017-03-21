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
from datetime import datetime
sk_seed = 0

# Some hyperparams
nb_epochs      = 2               # max training epochs
batch_size     = 32              # training batch size
max_arg_len    = 60              # max length of each arg
maxlen         = max_arg_len * 2 # max num of tokens per sample
num_units      = 4               # hidden layer size
num_layers     = 2               # try bidir?
max_time_steps = 100

###############################################################################
# Data
###############################################################################
conll_data = Data(
            max_arg_len=max_arg_len,
            maxlen=maxlen,
            split_input=True,
            bos_tag="<bos>",
            eos_tag="<eos>")

# X is a list of narrays: [arg1, arg2] , args are integers
# y is a numpy array: [samples x classes]
(X_train, classes_train, dec_train), (X_test, classes_test, dec_test) = \
                                                          conll_data.get_data()
# Encoder decoder inputs
x_train_enc, x_train_dec = X_train[0], X_train[1]
x_test_enc, x_test_dc = X_test[-1], X_test[1]

# Sequence length as numpy array shape [samples x 2]
seq_len_train, seq_len_val = conll_data.get_seq_length()
enc_len_train, dec_len_train = seq_len_train[:,0], seq_len_train[:,1]

# Decoder loss masking
# For mask to work, padding must be integer 0
train_dec_mask = np.sign(X_train[1])
test_dec_mask  = np.sign(X_test[1])

# Word embeddings
emb = Embeddings(conll_data.vocab, conll_data.inv_vocab)
# embedding is a numpy array [vocab size x embedding dimension]
embedding = emb.get_embedding_matrix(\
            model_path='data/google_news_300.bin',
            save=True,
            load_saved=True)

# TODO: For conditional enc/dec, concat embedding with y class

###############################################################################
# Main stuff
###############################################################################
model = BasicEncDec(\
        num_units=num_units,
        max_seq_len=max_arg_len,
        embedding=embedding)

def make_batches(data, batch_size, shuffle=True):
  """ Batches the passed data
  Args:
    data       : a list of numpy arrays
    batch_size : int
    shuffle    : should be true except when testing
  Returns:
    list of original numpy arrays but sliced
  """
  sk_seed = np.random.randint(0,10000)
  if shuffle: data = group_shuffle(*data, random_state=sk_seed)
  data_size = len(data[0])
  batch_per_epoch = int(data_size/batch_size) + 1
  for batch_num in range(batch_per_epoch):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, data_size)
    batch = []
    for d in data:
      batch.append(d[start_index:end_index])
    yield batch

with tf.Session() as sess:
  tf.global_variables_initializer().run()
  data = [x_train_enc, x_train_dec, enc_len_train, dec_len_train, dec_train,
          train_dec_mask]
  for epoch in range(nb_epochs):
    t1 = datetime.now()
    batches = make_batches(data, batch_size,shuffle=True)
    for batch in batches:
      b_train_enc, b_train_dec = batch[0], batch[1]
      b_enc_len_train, b_dec_len_train = batch[2], batch[3]
      b_dec_targets = batch[4]
      b_train_dec_mask = batch[5]
      fetch = [model.optimizer, model.cost]
      feed = {
               model.enc_input       : b_train_enc,
               model.enc_input_len   : b_enc_len_train,
               model.dec_targets     : b_dec_targets,
               model.dec_input       : b_train_dec,
               model.dec_input_len   : b_dec_len_train,
               model.dec_weight_mask : b_train_dec_mask
             }
      _, loss = sess.run(fetch,feed)

      diff_t = (datetime.now() - t1).total_seconds()
      print('epoch: {:2.0f} time: {:>4.1f} | loss: {:>3.4f} '.format(
        epoch, diff_t, loss), end='\r')



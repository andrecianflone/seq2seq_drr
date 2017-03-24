"""
Author: Andre Cianflone
Encoder Decoder type model for DRR
"""
from helper import Data
from embeddings import Embeddings
import tensorflow as tf
import numpy as np
from enc_dec import BasicEncDec
from utils import Progress, make_batches
sk_seed = 0

# Some hyperparams
nb_epochs      =  2              # max training epochs
batch_size     = 32              # training batch size
max_arg_len    = 60              # max length of each arg
maxlen         = max_arg_len * 2 # max num of tokens per sample
num_units      = 4               # hidden layer size
num_layers     = 2               # try bidir?
max_time_steps = 100

###############################################################################
# Data
###############################################################################
# TODO: data section is messy, needs some cleaning
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
x_test_enc, x_test_dec = X_test[0], X_test[1]

# Sequence length as numpy array shape [samples x 2]
seq_len_train, seq_len_test = conll_data.get_seq_length()
enc_len_train, dec_len_train = seq_len_train[:,0], seq_len_train[:,1]
enc_len_test, dec_len_test = seq_len_test[:,0], seq_len_test[:,1]

# Decoder loss masking
# For mask to work, padding must be integer 0
dec_mask_train = np.sign(X_train[1])
dec_mask_test  = np.sign(X_test[1])

# Word embeddings
emb = Embeddings(conll_data.vocab, conll_data.inv_vocab)
# embedding is a numpy array [vocab size x embedding dimension]
embedding = emb.get_embedding_matrix(\
            model_path='data/google_news_300.bin',
            save=True,
            load_saved=True)

num_batches_train = len(x_train_enc)//batch_size+(len(x_train_enc)%batch_size>0)
num_batches_test = len(x_test_enc)//batch_size+(len(x_test_enc)%batch_size>0)
###############################################################################
# Main stuff
###############################################################################
model = BasicEncDec(\
        num_units=num_units,
        max_seq_len=max_arg_len,
        embedding=embedding,
        num_classes=conll_data.num_classes,
        emb_dim=embedding.shape[1])

prog = Progress(batches=num_batches_train, progress_bar=True, bar_length=30)

def train_one_epoch():
  data = [x_train_enc, x_train_dec, classes_train, enc_len_train,
          dec_len_train, dec_train,dec_mask_train]
  batches = make_batches(data, batch_size, num_batches_train, shuffle=True)
  for batch in batches:
    b_enc         = batch[0]
    b_dec         = batch[1]
    b_classes     = batch[2]
    b_enc_len     = batch[3]
    b_dec_len     = batch[4]
    b_dec_targets = batch[5]
    b_dec_mask    = batch[6]
    fetch = [model.optimizer, model.cost]
    feed = {
             model.enc_input       : b_enc,
             model.enc_input_len   : b_enc_len,
             model.classes         : b_classes,
             model.dec_targets     : b_dec_targets,
             model.dec_input       : b_dec,
             model.dec_input_len   : b_dec_len,
             model.dec_weight_mask : b_dec_mask
           }
    _, loss = sess.run(fetch,feed)
    prog.print_train(loss)

def eval_test_set():
  data = [x_test_enc, x_test_dec, classes_test, enc_len_test,
          dec_len_test, dec_test, dec_mask_test]
  batches = make_batches(data, batch_size, num_batches_test, shuffle=True)
  losses = np.zeros(num_batches_test) # to average the losses
  batch_w = np.zeros(num_batches_test) # batch weight
  for i, batch in enumerate(batches):
    b_enc         = batch[0]
    b_dec         = batch[1]
    b_classes     = batch[2]
    b_enc_len     = batch[3]
    b_dec_len     = batch[4]
    b_dec_targets = batch[5]
    b_dec_mask    = batch[6]
    fetch = [model.cost]
    feed = {
             model.enc_input       : b_enc,
             model.enc_input_len   : b_enc_len,
             model.classes         : b_classes,
             model.dec_targets     : b_dec_targets,
             model.dec_input       : b_dec,
             model.dec_input_len   : b_dec_len,
             model.dec_weight_mask : b_dec_mask
           }
    loss = sess.run(fetch,feed)
    loss = loss[0]

    # Keep track of losses to average later
    cur_b_size = b_enc.shape[0]
    losses[i] = loss
    batch_w[i] = cur_b_size / len(x_test_enc)

  # Average across batches
  av = np.average(losses, weights=batch_w)
  prog.print_eval(av)

# Launch training
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for epoch in range(nb_epochs):
    prog.epoch_start()
    train_one_epoch()
    eval_test_set()
    prog.epoch_end()


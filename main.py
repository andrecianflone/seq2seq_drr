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
nb_epochs      = 20              # max training epochs
batch_size     = 32              # training batch size
max_arg_len    = 60              # max length of each arg
maxlen         = max_arg_len * 2 # max num of tokens per sample
cell_units     = 64             # hidden layer size
dec_out_units  = 32
num_layers     = 2               # try bidir?
max_time_steps = 100
keep_prob      = 0.3

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
        num_units=cell_units,
        dec_out_units=dec_out_units,
        max_seq_len=max_arg_len,
        embedding=embedding,
        num_classes=conll_data.num_classes,
        emb_dim=embedding.shape[1])

prog = Progress(batches=num_batches_train, progress_bar=True, bar_length=30)

def call_model(data, fetch, num_batches, keep_prob, shuffle):
  """ Calls models and yields results per batch """
  batches = make_batches(data, batch_size, num_batches, shuffle=shuffle)
  results = []
  for batch in batches:
    b_enc         = batch[0]
    b_dec         = batch[1]
    b_classes     = batch[2]
    b_enc_len     = batch[3]
    b_dec_len     = batch[4]
    b_dec_targets = batch[5]
    b_dec_mask    = batch[6]
    feed = {
             model.enc_input       : b_enc,
             model.enc_input_len   : b_enc_len,
             model.classes         : b_classes,
             model.dec_targets     : b_dec_targets,
             model.dec_input       : b_dec,
             model.dec_input_len   : b_dec_len,
             model.dec_weight_mask : b_dec_mask,
             model.keep_prob       : keep_prob
           }
    result = sess.run(fetch,feed)
    # yield the results when training
    yield result

def train_one_epoch():
  data = [x_train_enc, x_train_dec, classes_train, enc_len_train,
          dec_len_train, dec_train,dec_mask_train]
  fetch = [model.optimizer, model.cost]
  batch_results = call_model(data, fetch, num_batches_train, keep_prob, shuffle=True)
  for result in batch_results:
    loss = result[1]
    prog.print_train(loss)

def test_set_decoder_loss():
  """ Get the total loss for the entire batch """
  data = [x_test_enc, x_test_dec, classes_test, enc_len_test,
          dec_len_test, dec_test, dec_mask_test]
  fetch = [model.batch_size, model.cost]
  losses = np.zeros(num_batches_test) # to average the losses
  batch_w = np.zeros(num_batches_test) # batch weight
  batch_results = call_model(data, fetch, num_batches_test, keep_prob=1, shuffle=False)
  for i, result in enumerate(batch_results):
    # Keep track of losses to average later
    cur_b_size = result[0]
    losses[i] = result[1]
    batch_w[i] = cur_b_size / len(x_test_enc)

  # Average across batches
  av = np.average(losses, weights=batch_w)
  prog.print_dec_eval(av)

def test_set_classification_loss():
  """ Try all label conditioning for eval dataset
  For each sample, get the perplexity when conditioning on all classes and set
  the label with argmin. Check accuracy and f1 score of classification
  """
  # To average the losses
  log_prob = np.zeros((len(classes_test), conll_data.num_classes), dtype=np.float32)
  for k, v in conll_data.sense_to_one_hot.items(): # loop over all classes
    class_id = np.argmax(v)
    classes = np.array([v])
    classes = np.repeat(classes, len(classes_test), axis=0)
    assert classes_test.shape == classes.shape

    data = [x_test_enc, x_test_dec, classes, enc_len_test,
          dec_len_test, dec_test, dec_mask_test]
    fetch = [model.batch_size, model.generator_loss, model.softmax_logits,
              model.dec_targets]
    batch_results = call_model(data, fetch, num_batches_test, keep_prob=1, shuffle=False)
    j = 0
    for result in batch_results:
      cur_b_size = result[0]
      # loss = result[1]

      # Get the probability of the words we want
      targets = result[3]
      probs = result[2] # [batch_size, time step, vocab_size]
      targets = targets[:,0:probs.shape[1]] # ignore zero pads
      I,J=np.ix_(np.arange(probs.shape[0]),np.arange(probs.shape[1]))
      prob_vocab = probs[I,J,targets]
      # Get the sum log across all words per sample
      sum_log_prob = np.sum(np.log(prob_vocab), axis=1) # [batch_size,]
      # Assign the sum log prob to the correct class column
      log_prob[j:j+cur_b_size, class_id] = sum_log_prob
      j += cur_b_size

  predictions = np.argmax(log_prob, axis=1) # get index of most probable
  gold = np.argmax(classes_test, axis=1) # get index of one hot class
  correct = predictions == gold # compare how many match
  accuracy = np.sum(correct)/len(correct)
  prog.print_class_eval(accuracy)

# Launch training
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  for epoch in range(nb_epochs):
    prog.epoch_start()
    train_one_epoch()
    test_set_decoder_loss()
    # test_set_classification_loss()
    prog.epoch_end()


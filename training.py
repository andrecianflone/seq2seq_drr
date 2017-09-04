import tensorflow as tf
from helper import make_batches, MiniData
from utils import Progress, Metrics, Callback
import numpy as np
import sys
from pprint import pprint
from sklearn.metrics import f1_score, accuracy_score
from six.moves import cPickle as pickle

###############################################################################
# Training/Testing functions
###############################################################################
def call_model(sess, model, data, fetch, batch_size, num_batches, keep_prob,
              shuffle, mode):
  """ Calls models and yields results per batch """
  batches = make_batches(data, batch_size, num_batches, shuffle=shuffle)
  for batch in batches:
    feed = {
             model.enc_input       : batch.encoder_input,
             model.enc_input_len   : batch.seq_len_encoder,
             model.classes         : batch.classes,
             model.dec_targets     : batch.decoder_target,
             model.dec_input       : batch.decoder_input,
             model.dec_input_len   : batch.seq_len_decoder,
             model.keep_prob       : keep_prob,
             model.mode            : mode # 1 for train, 0 for testing
           }

    result = sess.run(fetch,feed)
    yield result

def generate_text(sess, model, data, index, vocab, inv_vocab):
  """
  Return a sample of text generated, dependent on encoder input, and index
  """
  ids = [index,index+1]
  sample = MiniData(data, ids)
  fetch = [model.sample_id]
  feed = {
           model.enc_input       : sample.encoder_input,
           model.enc_input_len   : sample.seq_len_encoder,
           model.dec_targets     : sample.decoder_target,
           model.dec_input       : sample.decoder_input,
           model.dec_input_len   : sample.seq_len_decoder,
           model.keep_prob       : 1,
           model.mode            : 0
         }
  result = sess.run(fetch,feed)
  predicted = result[0]

  # Swap for words
  encoded = ' '.join([inv_vocab[x] for x in sample.encoder_input[0,:]])
  decoded = ' '.join([inv_vocab[x] for x in predicted[0,:]])
  target  = ' '.join([inv_vocab[x] for x in sample.decoder_target[0,:]])
  relation = sample.classes[0]

  return relation, encoded, decoded, target

def train_one_epoch(sess, data, model, keep_prob, batch_size, num_batches,
                    prog, writer=None):
  """ Train 'model' using 'data' for a single epoch """
  fetch = [model.optimizer, model.cost, model.global_step]

  if writer is not None:
    fetch.append(model.merged_summary_ops)

  batch_results = call_model(sess, model, data, fetch, batch_size, num_batches,
                             keep_prob, shuffle=True, mode=1)
  for result in batch_results:
    loss = result[1]
    global_step = result[2]
    prog.print_train(loss)
    if writer is not None and global_step % 10 == 0:
      summary = result[-1]
      writer.add_summary(summary, global_step)
    # break

def classification_f1(sess, data, model, batch_size, num_batches_test, save_align):
  """
  Get the total loss for the entire batch
  Args:
    save_align: if true, will save alignment history to disk, from hparams
  """
  fetch = [model.batch_size, model.cost, model.y_pred, model.y_true]

  alignment_ls = []
  if save_align == True:
    fetch.append(model.enc_input)
    fetch.append(model.dec_input)
    fetch.append(model.alignment_history)

  y_pred = np.zeros(data.size())
  y_true = np.zeros(data.size())
  batch_results = call_model(sess, model, data, fetch, batch_size,
               num_batches_test, keep_prob=1, shuffle=False, mode=0)
  start_id = 0
  for i, result in enumerate(batch_results):
    batch_size                           = result[0]
    cost                                 = result[1]
    y_pred[start_id:start_id+batch_size] = result[2]
    y_true[start_id:start_id+batch_size] = result[3]
    start_id += batch_size
    if save_align == True:
      enc_in = result[4]
      dec_in = result[5]
      align  = result[6]
      al_ls = alignment(enc_in, dec_in, align, data_class.inv_vocab)
      alignment_ls.extend(al_ls)

  # Metrics
  # f1 score depending on number of classes
  if data.num_classes == 2:
    # If only 2 classes, then one is positive, and average is binary
    pos_label = np.argmax(data.sense_to_one_hot['positive'])
    average = 'binary'
    f1_micro = f1_score(y_true, y_pred, pos_label=pos_label, average='binary')
  else:
    # If multiclass, no positive labels
    f1_micro = f1_score(y_true, y_pred, average='micro')

  acc = accuracy_score(y_true, y_pred)
  # f1_conll = data_class.conll_f1_score(y_pred, data.orig_disc, data.path_source)
  f1_conll =f1_micro
  return f1_micro, f1_conll, acc, alignment_ls

def test_set_decoder_loss(sess, data, model, batch_size, num_batches):
  """ Get the total loss for the entire batch """
  fetch = [model.batch_size, model.cost]
  losses = np.zeros(num_batches) # to average the losses
  batch_w = np.zeros(num_batches) # batch weight
  batch_results = call_model(sess, model, data, fetch, batch_size, num_batches,
                              keep_prob=1, shuffle=False, mode=0)
  for i, result in enumerate(batch_results):
    # Keep track of losses to average later
    cur_b_size = result[0]
    losses[i] = result[1]
    batch_w[i] = cur_b_size / len(x_test_enc)

  # Average across batches
  av = np.average(losses, weights=batch_w)
  return av

def language_model_class_loss():
  """ Try all label conditioning for eval dataset
  For each sample, get the perplexity when conditioning on all classes and set
  the label with argmin. Check accuracy and f1 score of classification
  """
  # To average the losses
  log_prob = np.zeros((len(classes_test), data_class.num_classes), dtype=np.float32)
  for k, v in data_class.sense_to_one_hot.items(): # loop over all classes
    class_id = np.argmax(v)
    classes = np.array([v])
    classes = np.repeat(classes, len(classes_test), axis=0)
    assert classes_test.shape == classes.shape

    data = [x_test_enc, x_test_dec, classes, enc_len_test,
          dec_len_test, dec_test]
    fetch = [model.batch_size, model.generator_loss, model.softmax_logits,
              model.dec_targets]
    batch_results = call_model(sess, model, data, fetch, batch_size,
                 num_batches_test, keep_prob=1, shuffle=False, mode_train=False)
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

current_trial = 0
# Launch training
def train(params, settings, Model, embedding, emb_dim, dataset_dict, vocab, inv_vocab):
  global hparams
  hparams=params
  global current_trial
  current_trial += 1
  print('-' * 79)
  print('Current trial: {}'.format(current_trial))
  print('-' * 79)
  # tf.reset_default_graph() # reset the graph for each trial
  train_set = dataset_dict['training_set']
  val_set = dataset_dict['validation_set']
  prog = Progress(batches=train_set.num_batches(hparams.batch_size), progress_bar=True,
                  bar_length=10)

  # Print some info
  pprint(hparams)
  # Dataset info
  for name, dataset in dataset_dict.items():
    print('Size of {} : {}'.format(name, dataset.size()))

  # Save trials along the way
  # pickle.dump(trials, open("trials.p","wb"))

  # Declare model with hyperhparams
  with tf.Graph().as_default(), tf.Session() as sess:
    tf.set_random_seed(1)
    model = Model(hparams, embedding, emb_dim)

    # Save info for tensorboard
    if settings['tensorboard_write'] == True:
      writer = tf.summary.FileWriter('logs', sess.graph)
    else:
      writer = None

    # Initialize variables
    tf.global_variables_initializer().run()

    # trask specific training
    if model.model_type == "generative":
      train_generative(sess, hparams, prog, model,dataset_dict, vocab, inv_vocab)
    if model.model_type == "classification":
      train_classification(sess, hparams, prog, model,dataset_dict, vocab, inv_vocab)

def train_generative(sess, hparams, prog, model,dataset_dict, vocab, inv_vocab):
  train_set = dataset_dict['training_set']
  val_set = dataset_dict['validation_set']
  met = Metrics(monitor="loss")
  cb = Callback(hparams.early_stop_epoch, met, prog)

  # Prediction test
  relation, encoded, decoded, target = generate_text(sess, model, val_set, 9, vocab, inv_vocab)
  print('\nrelation: {}'.format(relation))
  print('encoded: {}'.format(encoded))
  print('target: {}'.format(target))
  for epoch in range(hparams.nb_epochs):
    prog.epoch_start()

    # Training set
    train_one_epoch(sess, train_set, model, hparams.keep_prob,
                hparams.batch_size, train_set.num_batches(hparams.batch_size), prog)

    # Test an output! See how it evolves!
    _, _, decoded, _ = generate_text(sess, model, val_set, 9, vocab, inv_vocab)
    print('\ndecoded: {}'.format(decoded),end='')

    # # Validation Set
    # prog.print_cust('|| {} '.format(val_set.short_name))
    # loss = test_set_decoder_loss(
            # sess, val_set, model, batch_size, val_set.num_batches(batch_size))
    # met.update(val_set.short_name + '_loss', loss)
    # prog.print_eval('loss', loss)

    # # Previous best f1 on test -> for alignment
    # prev_best = met.metric_dict["loss"]

    # for k, dataset in dataset_dict.items():
      # if k == "training_set": continue # skip training, already done
      # if k == "validation_set": continue # skip validation, already done

      # # Other sets
      # prog.print_cust('|| {} '.format(dataset.short_name))
      # loss = classification_f1(
          # sess, dataset, model, batch_size, dataset.num_batches(batch_size))
      # met.update(dataset.short_name + '_loss', loss)
      # prog.print_eval('loss', loss)

    # if cb.early_stop() == True: break
    prog.epoch_end()
  pass

def train_classification(sess, hparams, prog, model, dataset_dict, vocab,
                                                                    inv_vocab):
  train_set = dataset_dict['training_set']
  val_set = dataset_dict['validation_set']
  met = Metrics(monitor="val_f1")
  cb = Callback(hparams.early_stop_epoch, met, prog)
  align = False

  for epoch in range(hparams.nb_epochs):
    prog.epoch_start()

    # Training set
    train_one_epoch(sess, train_set, model, hparams.keep_prob,
          hparams.batch_size, train_set.num_batches(hparams.batch_size), prog)

    # Validation Set
    prog.print_cust('|| {} '.format(val_set.short_name))
    _, f1, accuracy, alignment = classification_f1(
        sess, val_set, model, hparams.batch_size,
        val_set.num_batches(hparams.batch_size), align)
    met.update(val_set.short_name + '_f1', f1)
    met.update(val_set.short_name + '_acc', accuracy)
    prog.print_eval('acc', accuracy)
    prog.print_eval('f1', f1)

    # Previous best f1 on test -> for alignment
    prev_best = met.metric_dict["test_f1"]

    for k, dataset in dataset_dict.items():
      if k == "training_set": continue # skip training, already done
      if k == "validation_set": continue # skip validation, already done

      # Other sets
      prog.print_cust('|| {} '.format(dataset.short_name))
      _, f1, accuracy, alignment = classification_f1(
          sess, dataset, model, hparams.batch_size,
          dataset.num_batches(hparams.batch_size), align)
      met.update(dataset.short_name + '_f1', f1)
      met.update(dataset.short_name + '_acc', accuracy)
      prog.print_eval('acc', accuracy)
      prog.print_eval('f1', f1)

      # if test set better, save alignment
      if align == True and k == "test_set":
        if prev_best < met.metric_dict["test_f1"]:
          # dump pickle
          pickle.dump(alignment, open("tmp.p", "wb"))
          print("dumped test alignments to tmp.p file")

    if cb.early_stop() == True: break
    prog.epoch_end()


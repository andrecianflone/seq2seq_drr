from tensorflow import tf
from helper import make_batches

###############################################################################
# Training/Testing functions
###############################################################################
def call_model(sess, model, data, fetch, batch_size, num_batches, keep_prob,
              shuffle, mode_train):
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
             model.dec_weight_mask : batch.decoder_mask,
             model.keep_prob       : keep_prob,
             model.mode_train      : mode_train
           }

    result = sess.run(fetch,feed)
    yield result

def train_one_epoch(sess, data, model, keep_prob, batch_size, num_batches,
                    prog, writer=None):
  """ Train 'model' using 'data' for a single epoch """
  fetch = [model.optimizer, model.cost, model.global_step]

  if writer is not None:
    fetch.append(model.merged_summary_ops)

  batch_results = call_model(sess, model, data, fetch, batch_size, num_batches,
                             keep_prob, shuffle=True, mode_train=True)
  for result in batch_results:
    loss = result[1]
    global_step = result[2]
    prog.print_train(loss)
    if writer is not None and global_step % 10 == 0:
      summary = result[-1]
      writer.add_summary(summary, global_step)
    # break

def classification_f1(sess, data, model, batch_size, num_batches_test):
  """ Get the total loss for the entire batch """
  fetch = [model.batch_size, model.class_cost, model.y_pred, model.y_true]

  alignment_history = True
  alignment_ls = []
  if alignment_history == True:
    fetch.append(model.enc_input)
    fetch.append(model.dec_input)
    fetch.append(model.alignment_history)

  y_pred = np.zeros(data.size())
  y_true = np.zeros(data.size())
  batch_results = call_model(sess, model, data, fetch, batch_size,
               num_batches_test, keep_prob=1, shuffle=False, mode_train=False)
  start_id = 0
  for i, result in enumerate(batch_results):
    batch_size                           = result[0]
    cost                                 = result[1]
    y_pred[start_id:start_id+batch_size] = result[2]
    y_true[start_id:start_id+batch_size] = result[3]
    start_id += batch_size
    if alignment_history == True:
      enc_in = result[4]
      dec_in = result[5]
      align  = result[6]
      al_ls = alignment(enc_in, dec_in, align, data_class.inv_vocab)
      alignment_ls.extend(al_ls)

  # Metrics
  # f1 score depending on number of classes
  if data_class.num_classes == 2:
    # If only 2 classes, then one is positive, and average is binary
    pos_label = np.argmax(data_class.sense_to_one_hot['positive'])
    average = 'binary'
    f1_micro = f1_score(y_true, y_pred, pos_label=pos_label, average='binary')
  else:
    # If multiclass, no positive labels
    f1_micro = f1_score(y_true, y_pred, average='micro')

  acc = accuracy_score(y_true, y_pred)
  # f1_conll = data_class.conll_f1_score(y_pred, data.orig_disc, data.path_source)
  f1_conll =f1_micro
  return f1_micro, f1_conll, acc, alignment_ls

def test_set_decoder_loss(sess, model, batch_size, num_batches, prog):
  """ Get the total loss for the entire batch """
  data = [x_test_enc, x_test_dec, classes_test, enc_len_test,
          dec_len_test, dec_test, dec_mask_test]
  fetch = [model.batch_size, model.class_cost]
  losses = np.zeros(num_batches_test) # to average the losses
  batch_w = np.zeros(num_batches_test) # batch weight
  batch_results = call_model(sess, model, data, fetch, batch_size, num_batches,
                              keep_prob=1, shuffle=False, mode_train=False)
  for i, result in enumerate(batch_results):
    # Keep track of losses to average later
    cur_b_size = result[0]
    losses[i] = result[1]
    batch_w[i] = cur_b_size / len(x_test_enc)

  # Average across batches
  av = np.average(losses, weights=batch_w)
  prog.print_eval('decoder loss', av)

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
          dec_len_test, dec_test, dec_mask_test]
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

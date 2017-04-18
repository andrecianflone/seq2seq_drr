import tensorflow as tf

from tensorflow.contrib.rnn import GRUCell, BasicLSTMCell, DropoutWrapper
from tensorflow.contrib.layers import xavier_initializer as glorot

class BasicEncDec():
  """ LSTM enc/dec as baseline, no attention """
  def __init__(self, num_units, dec_out_units, max_seq_len, embedding,
      num_classes, emb_dim):
    self.keep_prob = tf.placeholder(tf.float32)
    self.float_type = tf.float32
    self.int_type = tf.int32
    self.final_emb_dim = emb_dim + num_classes
    self.bi_encoder_hidden = num_units * 2
    decoder_num_units = num_units *2

    ############################
    # Model inputs
    ############################
    vocab_size = embedding.shape[0]
    # Embedding tensor is of shape [vocab_size x embedding_size]
    self.embedding_tensor = embedding

    # Encoder inputs
    self.enc_input = tf.placeholder(self.int_type, shape=[None, max_seq_len])
    self.enc_embedded = self.embedded(self.enc_input, self.embedding_tensor)
    self.enc_input_len = tf.placeholder(self.int_type, shape=[None,])

    # Class label
    self.classes = tf.placeholder(self.int_type, shape=[None, num_classes])

    # Condition on Y ==> Embeddings + labels
    # self.enc_embedded = self.emb_add_class(self.enc_embedded, self.classes)

    # Decoder inputs and targets
    self.dec_targets = tf.placeholder(self.int_type, shape=[None, max_seq_len])
    self.dec_input = tf.placeholder(self.int_type, shape=[None, max_seq_len])
    self.dec_embedded = self.embedded(self.dec_input, self.embedding_tensor)
    # self.dec_embedded = self.emb_add_class(self.dec_embedded, self.classes)
    self.dec_input_len = tf.placeholder(self.int_type, shape=[None,])
    # weight mask shape [batch_size x sequence_length]
    self.dec_weight_mask = tf.placeholder(self.float_type, shape=[None, max_seq_len])

    self.batch_size = tf.shape(self.enc_input)[0]

    ############################
    # Model (magic is here)
    ############################
    cell_enc_fw = GRUCell(num_units)
    cell_enc_fw = DropoutWrapper(cell_enc_fw, output_keep_prob=self.keep_prob)
    cell_enc_bw = GRUCell(num_units)
    cell_enc_bw = DropoutWrapper(cell_enc_bw, output_keep_prob=self.keep_prob)
    cell_enc = GRUCell(num_units)
    cell_enc = DropoutWrapper(cell_enc, output_keep_prob=self.keep_prob)

    cell_dec = GRUCell(decoder_num_units)
    cell_dec = DropoutWrapper(cell_dec, output_keep_prob=self.keep_prob)
    # should add second additional layer here

    # Get data from encoder: bidirectional
    self.encoded_outputs, self.encoded_state = self.encoder_bi(cell_enc_fw, \
                            cell_enc_bw, self.enc_embedded, self.enc_input_len)

    # Get data from encoder: unidirectional
    # self.encoded_outputs, self.encoded_state = self.encoder_one_way(cell_enc, \
                            # self.enc_embedded, self.enc_input_len)
    # Get data from decoder
    self.decoded_outputs, self.decoded_final_state = self.decoder_train_attn(
                            cell=cell_dec,
                            decoder_inputs=self.dec_embedded,
                            seq_len_enc=self.enc_input_len,
                            seq_len_dec=self.dec_input_len,
                            encoder_state=self.encoded_state,
                            attention_states=self.encoded_outputs,
                            mem_units=self.bi_encoder_hidden,
                            attn_units=dec_out_units)


    # CLASSIFICATION ##########
    # Output for classification, use last decoder hidden state
    self.class_logits = self.output_logits(
        self.decoded_final_state, dec_out_units, num_classes, "class_softmax")

    # Classification loss
    classes_max = tf.argmax(self.classes, axis=1)
    self.class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                      labels=classes_max,
                      logits=self.class_logits)

    # weights = tf.constant(self.weights_cross_entropy, dtype=self.float_type)
    # weights = tf.expand_dims(weights,0)
    # weights = tf.tile(weights, [self.batch_size,1])
    # class_cast = tf.cast(self.classes, dtype=self.float_type)
    # self.class_loss = tf.nn.weighted_cross_entropy_with_logits(
                      # targets=class_cast,
                      # logits=self.class_logits,
                      # pos_weight=weights)

    # self.class_loss = tf.nn.softmax_cross_entropy_with_logits(
                      # labels=self.classes,
                      # logits=self.class_logits)

    self.class_cost = tf.reduce_mean(self.class_loss) # average across batch
    self.class_optimizer = tf.train.AdamOptimizer(0.001).minimize(self.class_cost)

    self.y_pred, self.y_true = self.predict(self.class_logits, self.classes)
    # TODO: try class loss across sequence, but classifier only uses last step
    # in the sequence to label the argument.

    # GENERATION ##############
    # Outputs over vocab, for sequence
    self.seq_logits = self.sequence_output_logits(
                  self.decoded_outputs, dec_out_units, vocab_size)
    self.seq_softmax_logits = tf.nn.softmax(self.seq_logits)

    # Generator loss per sample
    self.gen_loss = self.sequence_loss(\
                        self.seq_logits, self.dec_targets, self.dec_weight_mask)
    self.gen_cost = tf.reduce_mean(self.gen_loss) # average across batch
    self.gen_optimizer = tf.train.AdamOptimizer(0.001).minimize(self.gen_cost)

  def embedded(self, word_ids, embedding_tensor, scope="embedding"):
    """Swap ints for dense embeddings, on cpu.
    word_ids correspond the proper row index of the embedding_tensor

    Args:
      words_ids: array of [batch_size x sequence of word ids]
      embedding_tensor: tensor from which to retrieve the embedding
    Returns:
      tensor of shape [batch_size, sequence length, embedding size]
    """
    with tf.variable_scope(scope):
      with tf.device("/cpu:0"):
        inputs = tf.nn.embedding_lookup(embedding_tensor, word_ids)
    return inputs

  def encoder_one_way(self, cell, x, seq_len, init_state=None, scope="encoder"):
    """ Dynamic encoder for one direction
    Returns:
      outputs: all sequence hidden states as Tensor of shape [batch,time,units]
      state: last hidden state
    """
    # Output is the outputs at all time steps, state is the last state
    with tf.variable_scope(scope):
      outputs, state = tf.nn.dynamic_rnn(\
                  cell, x, sequence_length=seq_len, initial_state=init_state,
                  dtype=self.float_type)
    # state is a StateTuple class with properties StateTuple.c and StateTuple.h
    return outputs, state

  def encoder_bi(self, cell_fw, cell_bw, x, seq_len, init_state_fw=None,
                  init_state_bw=None, scope="encoder"):
    """ Dynamic encoder for two directions
    Returns:
      outputs: a tuple(output_fw, output_bw), all sequence hidden states, each
               as tensor of shape [batch,time,units]
      state: tuple(output_state_fw, output_state_bw) containing the forward
             and the backward final states of bidirectional rnlast hidden state
    """
    # Output is the outputs at all time steps, state is the last state
    with tf.variable_scope(scope):
      outputs, state = tf.nn.bidirectional_dynamic_rnn(\
                  cell_fw=cell_fw,
                  cell_bw=cell_bw,
                  inputs=x,
                  sequence_length=seq_len,
                  initial_state_fw=init_state_fw,
                  initial_state_bw=init_state_bw,
                  dtype=self.float_type)
      # outputs: a tuple(output_fw, output_bw), all sequence hidden states,
      # each as tensor of shape [batch,time,units]
      # Since we don't need the outputs separate, we concat here
      outputs = tf.concat(outputs,2)
      outputs.set_shape([None, None, self.bi_encoder_hidden])
      state = tf.concat(state,1)
      state.set_shape([None, self.bi_encoder_hidden])
    return outputs, state

  def emb_add_class(self, enc_embedded, classes):
    """ Concatenate input and classes """

    num_classes = tf.shape(classes)[1]
    # final_emb_dim = tf.to_int32(tf.shape(enc_embedded)[2] + num_classes)
    time_steps = tf.shape(enc_embedded)[1]
    classes = tf.tile(classes, [1, time_steps]) # copy along axis=1 only
    classes = tf.reshape(classes, [-1, time_steps, num_classes]) # match input
    classes = tf.cast(classes, self.float_type)
    concat = tf.concat([enc_embedded, classes], 2) # concat 3rd dimension

    # Hardset the shape. This is hacky, but because of tf.reshape, it seems the
    # tensor loses it's shape property which causes problems with contrib.rnn
    # wich uses the shape property
    concat.set_shape([None, None, self.final_emb_dim])
    return concat

  def add_classes_to_state(self, state_tuple, classes):
    """ Concatenate hidden state with class labels
    Args:
      encoded_state: An LSTMStateTuple with properties c and h
      classes: one-hot encoded labels to be concatenated to StateTuple.h
    """
    # h is shape [batch_size, num_units]
    classes = tf.cast(classes, self.float_type)
    h_new = tf.concat([state_tuple.h, classes], 1) # concat along 1st axis
    new_state_tuple = tf.contrib.rnn.LSTMStateTuple(state_tuple.c, h_new)
    return new_state_tuple

  def decoder_train(self, cell, x, seq_len, encoder_state, scope="decoder"):
    """ Training decoder. Decoder initialized with passed state
    Returns:
      Tensor shaped [batch_size, max_time, cell.output_size] where max_time is
      the longest sequence in THIS batch, meaning the longest
      in sequence_length. May be shorter than *max*
    """
    with tf.variable_scope(scope):
      # Must specify a decoder function for training
      dec_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)

      # At every timestep in below, a slice is fed to the decoder_fn
      outputs, final_state, final_context_state = \
              tf.contrib.seq2seq.dynamic_rnn_decoder(\
              cell, dec_fn_train, inputs=x, sequence_length=seq_len)

    return outputs

  def decoder_train_attn(self, cell, decoder_inputs, seq_len_enc, seq_len_dec,
      encoder_state, attention_states, mem_units, attn_units):
    """
    see: https://www.tensorflow.org/versions/r1.1/api_guides/python/contrib.seq2seq#Attention
    Args:
      cell: an instance of RNNCell.
      x: decoder inputs for training
      seq_len_enc: seq. len. of encoder input, will ignore memories beyond seq len
      seq_len_dec: seq. len. of decoder input
      encoder_state: initial state for decoder
      attention_states: hidden states (from encoder) to attend over.
      mem_units: num of units in attention_states
      attn_units: depth of attention (output) tensor
    Returns:
      outputs.rnn_output : decoder hidden states at all time steps
      final_state.attention : last hidden state
    """

    # Attention Mechanisms. Bahdanau is additive style attention
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
        num_units = mem_units, # depth of query mechanism
        memory = attention_states, # hidden states to attend (output of RNN)
        memory_sequence_length=seq_len_enc, # masks false memories
        normalize=False, # normalize energy term
        name='BahdanauAttention')

    # Attention Wrapper: adds the attention mechanism to the cell
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell = cell,# Instance of RNNCell
        attention_mechanism = attn_mech, # Instance of AttentionMechanism
        attention_size = attn_units, # Int, depth of attention (output) tensor
        attention_history=False, # whether to store history in final output
        name="attention_wrapper")

    # TrainingHelper does no sampling, only uses inputs
    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs = decoder_inputs, # decoder inputs
        sequence_length = seq_len_dec, # decoder input length
        name = "decoder_training_helper")

    # Initial state for decoder
    # Clone attention state from current attention, but use encoder_state
    batch_size = tf.shape(decoder_inputs)[0]
    initial_state = attn_cell.zero_state(\
                    batch_size=batch_size, dtype=self.float_type)
    initial_state = initial_state.clone(cell_state = encoder_state)

    # Decoder setup
    decoder = tf.contrib.seq2seq.BasicDecoder(
              cell = attn_cell,
              helper = helper, # A Helper instance
              initial_state = initial_state, # initial state of decoder
              output_layer = None) # instance of tf.layers.Layer, like Dense

    # Perform dynamic decoding with decoder object
    # Outputs is a BasicDecoder object with properties rnn_output and sample_id
    outputs, final_state = tf.contrib.seq2seq.dynamic_decode(decoder)
    return outputs.rnn_output, final_state.attention

  def decoder_inference(self, cell, x, seq_len, encoder_state,bos_id, eos_id,
      max_seq, vocab_size, scope="inference"):
    """ Inference step decoding. Used to generate decoded text """
    with tf.variable_scope(scope):
      # TODO: check simple_decoder_fn_inference docs

      # Must specify a decoder function for inference
      dec_fn_inf = tf.contrib.seq2seq.simple_decoder_fn_inference(
          output_fn = self.output_logits,
          encoder_state = encodeR_state, # encoded state to initialize decoder
          embeddings = self.embedding_tensor, # embedding matrix
          start_of_sequence_id = bos_id, # bos tag ID of embedding matrix
          end_of_sequence_id = eos_id, # eos tag ID of embedding matrix
          maximum_length = max_seq,
          num_decoder_symbols = vocab_size,
          dtype=tf.int32,
          name="decoder_inf_func")

      # At evevery timestep in below, a slice is fed to the decoder_fn
      # Inputs is none, the input is inferred solely from decoder_fn
      # Outputs will be Tensor shaped [batch_size, max_time, cell.output_size]
      outputs, final_state, final_context_state = \
              tf.contrib.seq2seq.dynamic_rnn_decoder(\
              cell, dec_fn_train, inputs=None, sequence_length=None)
    return outputs

  def sequence_output_logits(self, decoded_outputs, num_units, vocab_size):
    """ Output projection over all timesteps
    Returns:
      logit tensor of shape [batch_size, timesteps, vocab_size]
    """
    # We need to get the sequence length for *this* batch, this will not be
    # equal for each batch since the decoder is dynamic. Meaning length is
    # equal to the longest sequence in the batch, not the max over data
    max_seq_len = tf.shape(decoded_outputs)[1]

    # Reshape to rank 2 tensor so timestep is no longer a dimension
    output = tf.reshape(decoded_outputs, [-1, num_units])

    # Get the logits
    logits = self.output_logits(output, num_units, vocab_size, "seq_softmax")

    # Reshape back to the original tensor shape
    logits = tf.reshape(logits, [-1, max_seq_len, vocab_size])
    return logits

  def output_logits(self, decoded_outputs, num_units, vocab_size, scope):
    """ Output projection function
    To be used for single timestep in RNN decoder
    """
    with tf.variable_scope(scope):
      w = tf.get_variable("weights", [num_units, vocab_size],
          dtype=self.float_type, initializer=glorot())
      b = tf.get_variable("biases", [vocab_size],
          dtype=self.float_type, initializer=tf.constant_initializer(0.0))

      logits = tf.matmul(decoded_outputs, w) + b
    return logits

  def sequence_loss(self, logits, targets, weight_mask):
    """ Loss on sequence, given logits and one-hot targets
    Default loss below is softmax cross ent on logits
    Arguments:
      logits : logits over predictions, [batch, seq_len, num_decoder_symb]
      targets : the class id, shape [batch_size, seq_len], dtype int
      weigth_mask : valid logits should have weight "1" and padding "0",
        [batch_size, seq_len] of dtype float
    """
    # TODO should not average_across_timesteps?
    # We need to delete zeroed elements in targets, beyond max sequence
    max_seq = tf.reduce_max(tf.reduce_sum(weight_mask, axis=1))
    max_seq = tf.to_int32(max_seq)
    # Slice time dimension to max_seq
    targets = tf.slice(targets, [0, 0], [-1, max_seq])
    weight_mask = tf.slice(weight_mask, [0,0], [-1, max_seq])
    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, weight_mask,
            average_across_batch=False)
    return loss

  def predict(self, pred_logits, classes):
    """ Returns class label (int) for prediction and gold
    Args:
      pred_logits : predicted logits, not yet softmax
      classes : labels as one-hot vectors
    """
    y_pred = tf.nn.softmax(pred_logits)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.argmax(classes, axis=1)

    return y_pred, y_true

  def log_prob(self, logits, targets):
    """ Calculate the perplexity of a sequence:
    \left(\prod_{i=1}^{N} \frac{1}{P(w_i|past)} \right)^{1/n}
    that is, the total product of 1 over the probability of each word, and n
    root of that total

    For language model, lower perplexity means better model
    """
    # Probability of entire vocabulary over time
    probs = tf.nn.softmax(logits)

    # Get the model probability of only the targets
    # Targets are the vocabulary index
    # probs = tf.gather(probs, targets)
    return probs


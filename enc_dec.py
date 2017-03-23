import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import xavier_initializer as glorot

class BasicEncDec():
  """ LSTM enc/dec as baseline, no attention """
  def __init__(self, num_units, max_seq_len, embedding, num_classes, emb_dim):
    self.keep_prob = tf.placeholder(tf.float32)
    self.float_type = tf.float32
    self.int_type = tf.int32
    self.final_emb_dim = emb_dim + num_classes

    ############################
    # Model inputs
    ############################
    vocab_size = embedding.shape[0]
    # Embedding tensor is of shape [vocab_size x embedding_size]
    self.embedding_tensor = embedding
    # self.embedding_tensor = tf.get_variable(
                        # name="embedding", shape=embedding.shape,
                        # initializer=tf.constant_initializer(embedding),
                        # trainable=False)

    # Encoder inputs
    self.enc_input = tf.placeholder(self.int_type, shape=[None, max_seq_len])
    self.enc_embedded = self.embedded(self.enc_input, self.embedding_tensor)
    self.enc_input_len = tf.placeholder(self.int_type, shape=[None,])

    # Class label
    self.classes = tf.placeholder(self.int_type, shape=[None, num_classes])
    # Condition on Y ==> Embeddings + labels
    self.enc_embedded = self.emb_add_class(self.enc_embedded, self.classes)

    # Decoder inputs and targets
    self.dec_targets = tf.placeholder(self.int_type, shape=[None, max_seq_len])
    self.dec_input = tf.placeholder(self.int_type, shape=[None, max_seq_len])
    self.dec_embedded = self.embedded(self.dec_input, self.embedding_tensor)
    self.dec_input_len = tf.placeholder(self.int_type, shape=[None,])
    # weight mask shape [batch_size x sequence_length]
    self.dec_weight_mask = tf.placeholder(self.float_type, shape=[None, max_seq_len])

    batch_size = tf.shape(self.enc_input)[0]

    ############################
    # Model
    ############################
    cell = BasicLSTMCell(num_units, state_is_tuple=True)
    # cell = DropoutWrapper(cell, output_keep_prob=keep_prob)
    # should add second additional layer here

    init_state = cell.zero_state(batch_size, self.float_type)
    # \begin{magic}
    self.encoded_state = self.encoder(cell, self.enc_embedded, self.enc_input_len,
                    init_state)
    # self.encoded_state = self.add_classes_to_state(self.encoded_state, self.classes)
    self.decoded_outputs = self.decoder_train(cell, self.dec_embedded,
                      self.dec_input_len, self.encoded_state)
    self.logits = self.out_logits(self.decoded_outputs, num_units, vocab_size)
    # \end{magic}
    self.generator_loss = \
            self.get_loss(self.logits, self.dec_targets, self.dec_weight_mask)
    self.cost = tf.reduce_sum(self.generator_loss)
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

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

  def encoder(self, cell, x, seq_len, init_state, scope="encoder"):
    """ Encodes input, returns last state"""
    # Output is the outputs at all time steps, state is the last state
    with tf.variable_scope(scope):
      output, state = tf.nn.dynamic_rnn(\
                  cell, x, sequence_length=seq_len, initial_state=init_state)
    # state is a StateTuple class with properties StateTuple.c and StateTuple.h
    return state

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
    # wich infers the shape
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
    """ Training decoder. Decoder initialized with passed state """
    with tf.variable_scope(scope):
      # Must specify a decoder function for training
      dec_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)

      # At every timestep in below, a slice is fed to the decoder_fn
      outputs, final_state, final_context_state = \
              tf.contrib.seq2seq.dynamic_rnn_decoder(\
              cell, dec_fn_train, inputs=x, sequence_length=seq_len)

    # Outputs will be Tensor shaped [batch_size, max_time, cell.output_size]
    # Max_time is the longest sequence in THIS batch, meaning the longest
    # in sequence_length. May be shorter than *max*
    return outputs

  def decoder_inference(self, scope="inference"):
    """ Inference step decoding."""
    with tf.variable_scope(scope):
      # Must specify a decoder function for inference
      dec_fn_inf = tf.contrib.seq2seq.simple_decoder_fn_inference()

      # At evevery timestep in below, a slice is fed to the decoder_fn
      # Inputs is none, the input is inferred solely from decoder_fn
      # Outputs will be Tensor shaped [batch_size, max_time, cell.output_size]
      outputs, final_state, final_context_state = \
              tf.contrib.seq2seq.dynamic_rnn_decoder(\
              cell, dec_fn_train, inputs=None, sequence_length=None)
    return outputs

  def out_logits(self, decoded_outputs, num_units, vocab_size):
    """ Softmax over decoder timestep outputs states """

    # We need to get the sequence length for *this* batch, this will not be
    # equal for each batch since the decoder is dynamic. Meaning length is
    # equal to the longest sequence in batch, not the max
    max_seq_len = tf.shape(decoded_outputs)[1]

    with tf.variable_scope("softmax"):
      w = tf.get_variable("weights", [num_units, vocab_size],
          dtype=self.float_type, initializer=glorot())
      b = tf.get_variable("biases", [vocab_size],
          dtype=self.float_type, initializer=tf.constant_initializer(0.0))

      # We need to reshape so timestep is no longer a dimension of output
      output = tf.reshape(decoded_outputs, [-1, num_units])
      logits = tf.matmul(output, w) + b
      # Get back original shape
      logits = tf.reshape(logits, [-1, max_seq_len, vocab_size])
    return logits

  def get_loss(self, logits, targets, weight_mask):
    """ Loss on sequence, given logits and one-hot targets
    Default loss below is softmax cross ent on logits
    Arguments:
      logits : logits over predictions, [batch, seq_len, num_decoder_symb]
      targets : the class id, shape [batch_size, seq_len], dtype int
      weigth_mask : valid logits should have weight "1" and padding "0",
        [batch_size, seq_len] of dtype float
    """
    # We need to delete zeroed elements in targets, beyond max sequence
    max_seq = tf.reduce_max(tf.reduce_sum(weight_mask, axis=1))
    max_seq = tf.to_int32(max_seq)
    # Slice time dimension to max_seq
    targets = tf.slice(targets, [0, 0], [-1, max_seq])
    weight_mask = tf.slice(weight_mask, [0,0], [-1, max_seq])
    return tf.contrib.seq2seq.sequence_loss(logits, targets, weight_mask)

  def predict(self):
    """ In inference step, must predict one step at a time. Beam search? """
    pass

  def perplexity(self):
    """ Calculate the perplexity of a sequence:
    \left(\prod_{i=1}^{N} \frac{1}{P(w_i|past)} \right)^{1/n}
    that is, the total product of 1 over the probability of each word, and n
    root of that total

    For language model, lower perplexity means better model
    """
    pass

''' Things to think about

################
Encoders/Decoders
################
# Different encoders
tf.nn.dynamic_rnn(
tf.nn.bidirectional_dynamic_rnn(
tf.contrib.rnn.stack_bidirectional_dynamic_rnn(

-------------------------------
# An idea from TensorFlow Dev conf
# https://www.youtube.com/watch?v=RIR_-Xlbp7s&list=PLOU2XLYxmsIKGc_NBoIhTn2Qhraji53cv&index=15
# RNN encoder via Fully Dynamic RNN
# 8 layer LSTM with residual connections, each layer on separate GPU, hence
the DeviceWrapper. Since you're stacking RNNs, you pass to MultiRNNCell
cell = MultiRNNCell(
        [DeviceWrapper(ResidualWrapper(LSTMCell(num_units=512)),
            device='/gpu:%d' % i)
        for i in range(8)])

encoder_outputs, encoder_final_state = dynamic_rnn(
        cell, inputs, sequence_length, parallel_iterations=32,
        swap_memory=True)
-------------------------------

################
Training stuff
################
batching with dynamic padding:
tf.train.batch(... dynamic_pad=True)

or have similar length sequences grouped together:
tf.contrib.training.bucket_by_sequence_length(... dynamic_pad=True)

We can automatically trunc sequences for BPTT with a state saver
tf.contrib.training.batch_sequences_with_states(...)

# trainer? Pipeline?
helper = TrainingHelper(decoder_inputs, sequence_length)
'''

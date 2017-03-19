import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import xavier_initializer as glorot

class BasicEncDec():
  """ LSTM enc/dec as baseline, no attention """
  def __init__(self, num_units, max_seq_len, embedding):
    keep_prob = tf.placeholder(tf.float32)
    self.data_type = tf.float32

    vocab_size = embedding.shape[0]
    # Embedding tensor is of shape [vocab_size x embedding_size]
    embedding_tensor = tf.get_variable(
                        name="embedding", shape=embedding.shape,
                        initializer=tf.constant_initializer(embedding),
                        trainable=False)

    enc_input = tf.placeholder(self.data_type, shape=[None, max_seq_len])
    enc_input = self.embedded(enc_input, embedding_tensor)
    enc_input_len = tf.placeholder(self.data_type, shape=[None, max_seq_len])

    dec_input = tf.placeholder(self.data_type, shape=[None, max_seq_len])
    dec_input = self.embedded(dec_input, embedding_tensor)
    dec_input_len = tf.placeholder(self.data_type, shape=[None, max_seq_len])
    # weight mask shape [batch_size x sequence_length]
    dec_input_weight_mask = tf.placeholder(self.data_type, shape=[None, max_seq_len])

    batch_size = tf.shape(enc_input.x)[0]

    cell = BasicLSTMCell(num_units, state_is_tuple=True)
    # cell = DropoutWrapper(cell, output_keep_prob=keep_prob)
    # should add second additional layer here

    # initial_state = cell.zero_state(batch_size, data_type())
    # \begin{magic}
    encoded_state = self.encoder(cell, enc_input, enc_input_len)
    decoded_outputs = self.decoder_train(dec_input, dec_input_len, encoded_state)
    logits = self.out_logits(decoded_outputs, num_units, max_seq_len, vocab_size)
    # \end{magic}
    loss = self.get_loss(logits, targets, dec_input_weight_mask)
    self.cost = tf.reduce_sum(loss)
    self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    def embedded(self, word_ids, embedding_tensor):
      """Swap ints for dense embeddings, on cpu.
      word_ids correspond the proper row index of the embedding_tensor

      Args:
        words_ids : array of [batch_size x sequence of word ids]
        embedding_tensor : tensor from which to retrieve the embedding
      Returns:
        tensor of shape [batch_size, sequence length, embedding size]
      """
      with tf.device("/cpu:0"):
        inputs = tf.nn.embedding_lookup(embedding_tensor, word_ids)
      return inputs

    def encoder(self, cell, x, seq_len):
      """ Encodes input, returns last state"""
      # Output is the outputs at all time steps, state is the last hidden state
      output, state = tf.nn.dynamic_rnn(\
                    cell, x, sequence_length=seq_len)
      return state

    def decoder_train(self, x, seq_len, encoder_state):
      """ Training decoder. Decoder initialized with passed state """
      # Must specify a decoder function for training
      dec_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)

      # At evevery timestep in below, a slice is fed to the decoder_fn
      outputs, final_state, final_context_state = \
                tf.contrib.seq2seq.dynamic_rnn_decoder(\
                cell, dec_fn_train, inputs=x, sequence_length=seq_len)

      # Outputs will be Tensor shaped [batch_size, max_time, cell.output_size]
      return outputs

    def decoder_inference(self):
      """ Inference step decoding."""
      # Must specify a decoder function for inference
      dec_fn_inf = tf.contrib.seq2seq.simple_decoder_fn_inference()

      # At evevery timestep in below, a slice is fed to the decoder_fn
      # Inputs is none, the input is inferred solely from decoder_fn
      # Outputs will be Tensor shaped [batch_size, max_time, cell.output_size]
      outputs, final_state, final_context_state = \
                tf.contrib.seq2seq.dynamic_rnn_decoder(\
                cell, dec_fn_train, inputs=None, sequence_length=None)
      return outputs

    def out_logits(self, decoded_outputs, num_units, max_seq_len, vocab_size):
      """ Softmax over decoder timestep outputs states """
      with tf.variable_scope("softmax"):
        w = tf.get_variable("weights", [num_units, vocab_size],
            dtype=data_type(), initializer=glorot())
        b = tf.get_variable("biases", [vocab_size],
            dtype=data_type(), initializer=tf.constant_initializer(0.0))

        # We need to reshape so timestep is no longer a dimension of output
        output = tf.reshape(decoded_outputs, [-1, num_units])
        logits = tf.matmul(output, w) + b
        # Get back original shape
        logits = tf.reshape(logits, [-1, max_seq_len, vocab_size])

    def get_loss(self, logits, targets, weight_mask):
      """ Loss on sequence, given logits and one-hot targets
      Default loss below is softmax cross ent on logits
      Arguments:
        logits : logits over predictions, [batch, seq_len, num_decoder_symb]
        targets : the class id, shape [batch_size, seq_len], dtype int
        weigth_mask : valid logits should have weight "1" and padding "0",
          [batch_size, seq_len] of dtype float
      """
      return tf.contrib.seq2seq.sequence_loss(logits, targets, weights_mask)

    def predict(self):
      # TODO: simply argmax over logits
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

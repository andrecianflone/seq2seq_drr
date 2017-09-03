import tensorflow as tf

from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.layers import xavier_initializer as glorot
from utils import dense
from pydoc import locate

class EncDec():
  """ Encoder Decoder """
  def __init__(self,params, embedding,emb_dim, num_classes=None, output_layer=None):
    """
    Args:
      hparams: hyper param instance
      embedding : embedding matrix as numpy array
      emb_dim : size of an embedding
    """
    global hparams
    hparams = params
    self.num_classes = num_classes
    self.floatX = tf.float32
    self.intX = tf.int32

    # self.final_emb_dim = emb_dim + num_classes
    self.bi_encoder_hidden = hparams.cell_units * 2
    if hparams.bidirectional == True:
      decoder_num_units = self.bi_encoder_hidden # double if bidirectional
    else:
      decoder_num_units = hparams.cell_units

    # helper variable to keep track of steps
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    ############################
    # Inputs
    ############################
    self.keep_prob = tf.placeholder(self.floatX)
    self.mode = tf.placeholder(tf.bool, name="mode") # 1 stands for training
    # self.max_infer_len = tf.placeholder(tf.intX) # max steps in inferences
    self.vocab_size = embedding.shape[0]
    # Embedding tensor is of shape [vocab_size x embedding_size]
    self.embedding_tensor = self.embedding_setup(embedding, hparams.emb_trainable)

    # Encoder inputs
    with tf.name_scope("encoder_input"):
      self.enc_input = tf.placeholder(self.intX, shape=[None, hparams.max_seq_len])
      enc_embedded = self.embedded(self.enc_input, self.embedding_tensor)
      self.enc_embedded = tf.layers.batch_normalization(enc_embedded, training=self.mode)
      self.enc_input_len = tf.placeholder(self.intX, shape=[None,])

    # Condition on Y ==> Embeddings + labels
    # self.enc_embedded = self.emb_add_class(self.enc_embedded, self.classes)

    # Decoder inputs and targets
    with tf.name_scope("decoder_input"):
      self.dec_targets = tf.placeholder(self.intX, shape=[None, hparams.max_seq_len])
      self.dec_input = tf.placeholder(self.intX, shape=[None, hparams.max_seq_len])
      dec_embedded = self.embedded(self.dec_input, self.embedding_tensor)
      self.dec_embedded = tf.layers.batch_normalization(dec_embedded, training=self.mode)
      # self.dec_embedded = self.emb_add_class(self.dec_embedded, self.classes)
      self.dec_input_len = tf.placeholder(self.intX, shape=[None,])
      # weight mask shape [batch_size x sequence_length]
      self.dec_weight_mask = tf.placeholder(self.floatX, shape=[None, hparams.max_seq_len])

    self.batch_size = tf.shape(self.enc_input)[0]

    ############################
    # Build Model
    ############################
    # Setup cells
    cell_enc_fw, cell_enc_bw, cell_enc, cell_dec = \
        self._build_cell(hparams.cell_units, decoder_num_units, cell_type=hparams.cell_type)

    # Get encoder data
    with tf.name_scope("encoder"):
      if hparams.bidirectional == True:
        self.encoded_outputs, self.encoded_state = self.encoder_bi(cell_enc_fw, \
                              cell_enc_bw, self.enc_embedded, self.enc_input_len)
      else:
        self.encoded_outputs, self.encoded_state = self.encoder_one_way(\
                              cell_enc, self.enc_embedded, self.enc_input_len)

    # Get decoder data
    with tf.name_scope("decoder"):
      # Get attention
      self.attn_cell, self.initial_state = self.decoder_attn(
                            self.batch_size,
                            cell=cell_dec,
                            mem_units=self.bi_encoder_hidden,
                            attention_states=self.encoded_outputs,
                            seq_len_enc=self.enc_input_len,
                            attn_units=hparams.dec_out_units,
                            encoder_state=self.encoded_state)

      self.decoded_outputs, self.decoded_final_state, self.decoded_final_seq_len=\
                      self.decoder_train(
                            self.batch_size,
                            attn_cell=self.attn_cell,
                            initial_state=self.initial_state,
                            decoder_inputs=self.dec_embedded,
                            seq_len_dec=self.dec_input_len,
                            output_layer=output_layer)

    self.alignment_history = self.decoded_final_state.alignment_history.stack()

    # Merged summary ops
    self.merged_summary_ops = tf.summary.merge_all()

  def get_optimizer(self, l_rate):
    Opt = locate("tensorflow.train." + hparams.optimizer)
    if Opt is None:
      raise ValueError("Invalid optimizer: " + hparams.optimizer)
    return Opt(l_rate)

  def embedding_setup(self, embedding, emb_trainable):
    """ If trainable, returns variable, otherwise the original embedding """
    if emb_trainable == True:
      emb_variable = tf.get_variable(
          name="embedding_matrix", shape=embedding.shape,
          initializer = tf.constant_initializer(embedding))
      return emb_variable
    else:
      return embedding

  def embedded(self, word_ids, embedding_tensor, scope="embedding"):
    """Swap ints for dense embeddings, on cpu.
    word_ids correspond the proper row index of the embedding_tensor

    Args:
      words_ids: array of [batch_size x sequence of word ids]
      embedding_tensor: tensor from which to retrieve the embedding, word id
        takes corresponding tensor row
    Returns:
      tensor of shape [batch_size, sequence length, embedding size]
    """
    with tf.variable_scope(scope):
      with tf.device("/cpu:0"):
        inputs = tf.nn.embedding_lookup(embedding_tensor, word_ids)
    return inputs

  def _build_cell(self, num_units, decoder_num_units, cell_type="LSTMCell"):
    Cell = locate("tensorflow.contrib.rnn." + cell_type)
    if Cell is None:
      raise ValueError("Invalid cell type " + cell_type)
    cell_enc_fw = Cell(num_units)
    cell_enc_bw = Cell(num_units)
    cell_enc = Cell(num_units)
    cell_dec = Cell(decoder_num_units)

    # Dropout wrapper
    cell_enc_fw = DropoutWrapper(cell_enc_fw, output_keep_prob=self.keep_prob)
    cell_enc_bw = DropoutWrapper(cell_enc_bw, output_keep_prob=self.keep_prob)
    cell_enc = DropoutWrapper(cell_enc, output_keep_prob=self.keep_prob)
    cell_dec = DropoutWrapper(cell_dec, output_keep_prob=self.keep_prob)
    return cell_enc_fw, cell_enc_bw, cell_enc, cell_dec

  def encoder_one_way(self, cell, x, seq_len, init_state=None):
    """ Dynamic encoder for one direction
    Returns:
      outputs: all sequence hidden states as Tensor of shape [batch,time,units]
      state: last hidden state
    """
    # Output is the outputs at all time steps, state is the last state
    with tf.variable_scope("dynamic_rnn"):
      outputs, state = tf.nn.dynamic_rnn(\
                  cell, x, sequence_length=seq_len, initial_state=init_state,
                  dtype=self.floatX)
    # state is a StateTuple class with properties StateTuple.c and StateTuple.h
    return outputs, state

  def encoder_bi(self, cell_fw, cell_bw, x, seq_len, init_state_fw=None,
                  init_state_bw=None):
    """ Dynamic encoder for two directions
    Returns:
      outputs: a tuple(output_fw, output_bw), all sequence hidden states, each
               as tensor of shape [batch,time,units]
      state: tuple(output_state_fw, output_state_bw) containing the forward
             and the backward final states of bidirectional rnlast hidden state
    """
    # Output is the outputs at all time steps, state is the last state
    with tf.variable_scope("bidirectional_dynamic_rnn"):
      outputs, state = tf.nn.bidirectional_dynamic_rnn(\
                  cell_fw=cell_fw,
                  cell_bw=cell_bw,
                  inputs=x,
                  sequence_length=seq_len,
                  initial_state_fw=init_state_fw,
                  initial_state_bw=init_state_bw,
                  dtype=self.floatX)
      # outputs: a tuple(output_fw, output_bw), all sequence hidden states,
      # each as tensor of shape [batch,time,units]
      # Since we don't need the outputs separate, we concat here
      outputs = tf.concat(outputs,2)
      outputs.set_shape([None, None, self.bi_encoder_hidden])
      # If LSTM cell, then "state" is not a tuple of Tensors but an
      # LSTMStateTuple of "c" and "h". Need to concat separately then new
      if "LSTMStateTuple" in str(type(state[0])):
        c = tf.concat([state[0][0],state[1][0]],axis=1)
        h = tf.concat([state[0][1],state[1][1]],axis=1)
        state = tf.contrib.rnn.LSTMStateTuple(c,h)
      else:
        state = tf.concat(state,1)
        # Manually set shape to Tensor or all hell breaks loose
        state.set_shape([None, self.bi_encoder_hidden])
    return outputs, state

  def emb_add_class(self, enc_embedded, classes):
    """ Concatenate input and classes. Do not use for classification """

    num_classes = tf.shape(classes)[1]
    # final_emb_dim = tf.to_int32(tf.shape(enc_embedded)[2] + num_classes)
    time_steps = tf.shape(enc_embedded)[1]
    classes = tf.tile(classes, [1, time_steps]) # copy along axis=1 only
    classes = tf.reshape(classes, [-1, time_steps, num_classes]) # match input
    classes = tf.cast(classes, self.floatX)
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
    classes = tf.cast(classes, self.floatX)
    h_new = tf.concat([state_tuple.h, classes], 1) # concat along 1st axis
    new_state_tuple = tf.contrib.rnn.LSTMStateTuple(state_tuple.c, h_new)
    return new_state_tuple

  def decoder_attn(self, batch_size, cell, mem_units, attention_states,
      seq_len_enc, attn_units, encoder_state):
    """
    Args:
      cell: an instance of RNNCell.
      mem_units: num of units in attention_states
      attention_states: hidden states (from encoder) to attend over.
      seq_len_dec: seq. len. of decoder input
      attn_units: depth of attention (output) tensor
      encoder_state: initial state for decoder

    """

    # Attention Mechanisms. Bahdanau is additive style attention
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(
        num_units = mem_units, # depth of query mechanism
        memory = attention_states, # hidden states to attend (output of RNN)
        memory_sequence_length=seq_len_enc, # masks false memories
        normalize=True, # normalize energy term
        name='BahdanauAttention')

    # Attention Wrapper: adds the attention mechanism to the cell
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
        cell = cell,# Instance of RNNCell
        attention_mechanism = attn_mech, # Instance of AttentionMechanism
        attention_layer_size = attn_units, # Int, depth of attention (output) tensor
        alignment_history = True, # whether to store history in final output
        name="attention_wrapper")

    # Initial state for decoder
    # Clone attention state from current attention, but use encoder_state
    initial_state = attn_cell.zero_state(\
                    batch_size=batch_size, dtype=self.floatX)
    initial_state = initial_state.clone(cell_state = encoder_state)

    return attn_cell, initial_state

  def decoder_train(self, batch_size, attn_cell, initial_state, decoder_inputs,
      seq_len_dec, output_layer=None):
    """
    Args:
      attn_cell: cell wrapped with attention
      initial_state: initial_state for decoder
      decoder_inputs: decoder inputs for training
      seq_len_enc: seq. len. of encoder input, will ignore memories beyond this
      output_layer: Dense layer to project output units to vocab

    Returns:
      outputs: a BasicDecoderOutput with properties:
        rnn_output: outputs across time,
          if output_layer, then [batch_size,dec_seq_len, out_size]
          otherwise output is [batch_size,dec_seq_len, cell_num_units]
        sample_id: an argmax over time of rnn_output, Tensor of shape
          [batch_size, dec_seq_len]
      final_state: an AttentionWrapperState, a namedtuple which contains:
        cell_state: such as LSTMStateTuple
        attention: attention emitted at previous time step
        time: current time step (the last one)
        alignments: Tensor of alignments emitted at previous time step for
          each attention mechanism
        alignment_history: TensorArray of laignment matrices from all time
          steps for each attention mechanism. Call stack() on each to convert
          to Tensor
    """

    # TrainingHelper does no sampling, only uses sequence inputs
    helper = tf.contrib.seq2seq.TrainingHelper(
        inputs = decoder_inputs, # decoder inputs
        sequence_length = seq_len_dec, # decoder input length
        name = "decoder_training_helper")

    # Decoder setup. This decoder takes inputs and states and feeds it to the
    # RNN cell at every timestep
    decoder = tf.contrib.seq2seq.BasicDecoder(
          cell = attn_cell,
          helper = helper, # A Helper instance
          initial_state = initial_state, # initial state of decoder
          output_layer = output_layer) # instance of tf.layers.Layer, like Dense

    # Perform dynamic decoding with decoder object
    # If impute_fnished=True ensures finished states are copied through,
    # corresponding outputs are zeroed out. For proper backprop
    # Maximum iterations: should be fixed for training, different value for generation
    outputs, final_state, final_sequence_lengths= \
              tf.contrib.seq2seq.dynamic_decode(\
                decoder=decoder,
                impute_finished=True,
                maximum_iterations=hparams.max_seq_len) # if None, decode till decoder is done
    return outputs, final_state, final_sequence_lengths

  def output_logits(self, decoded_outputs, num_units, vocab_size, scope):
    """ Output projection function
    To be used for single timestep in RNN decoder
    """
    with tf.variable_scope(scope):
      w = tf.get_variable("weights", [num_units, vocab_size],
          dtype=self.floatX, initializer=glorot())
      b = tf.get_variable("biases", [vocab_size],
          dtype=self.floatX, initializer=tf.constant_initializer(0.0))

      logits = tf.matmul(decoded_outputs, w) + b
    return logits

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

class EncDecClass(EncDec):
  """
  EncDec for classification. Classification based on last decoded hidden state.
  To use, must provide encoder/decoder inputs + class label
  """
  def __init__(self, hparams, embedding, emb_dim):
    super().__init__(hparams, embedding, emb_dim, output_layer=None)

    # Class label
    with tf.name_scope("class_labels"):
      # Labels for classification, single label per sample
      self.classes = tf.placeholder(self.intX, shape=[None, num_classes])

    with tf.name_scope("classification"):
      if hparams.class_over_sequence == True:
        # Classification over entire sequence output
        self.class_logits = self.sequence_class_logits(\
            decoded_outputs=self.decoded_outputs,
            pool_size=hparams.dec_out_units,
            max_seq_len=hparams.max_seq_len,
            num_classes=num_classes)
      else:
        # Classification input uses only sequence final state
        self.class_logits = self.output_logits(self.decoded_final_state.attention,
                            hparams.dec_out_units, num_classes, "class_softmax")

    # Classification loss
    self.loss = self.classification_loss(self.classes, self.class_logits)

    self.cost = tf.reduce_mean(self.loss) # average across batch

    tf.summary.scalar("class_cost", self.cost)

    self.y_pred, self.y_true = self.predict(self.class_logits, self.classes)

    # Loss ###################
    self.optimizer = self.get_optimizer(hparams.l_rate).minimize(\
                                      self.cost, global_step=self.global_step)

  def sequence_class_logits(self, decoded_outputs, pool_size, max_seq_len, num_classes):
    """ Logits for the sequence """
    with tf.variable_scope("pooling"):
      features = tf.expand_dims(self.decoded_outputs.rnn_output, axis=-1)
      pooled = tf.nn.max_pool(
          value=features, # [batch, height, width, channels]
          ksize=[1, 1, pool_size, 1],
          strides=[1, 1, 1, 1],
          padding='VALID',
          name="pool")
      # Get rid of last 2 empty dimensions
      pooled = tf.squeeze(pooled, axis=[2,3], name="pool_squeeze")
      # Pad
      pad_len = max_seq_len - tf.shape(pooled)[1]
      paddings = [[0,0],[0, pad_len]]
      x = tf.pad(pooled, paddings=paddings, mode='CONSTANT', name="padding")

    with tf.variable_scope("dense_layers"):
      # FC layers
      out_dim = hparams.hidden_size
      in_dim=max_seq_len
      for i in range(0,self.hparams.fc_num_layers):
        layer_name = "fc_{}".format(i+1)
        x = dense(x, in_dim, out_dim, act=tf.nn.relu, scope=layer_name)
        x = tf.nn.dropout(x, self.keep_prob)
        in_dim=out_dim

    # Logits
    logits = dense(x, out_dim, num_classes, act=None, scope="class_log")
    return logits

  def classification_loss(self, classes_true, classes_logits):
    """ Class loss. If binary, two outputs"""
    entropy_fn = tf.nn.sparse_softmax_cross_entropy_with_logits

    classes_max = tf.argmax(classes_true, axis=1)
    class_loss = entropy_fn(
                      labels=classes_max,
                      logits=classes_logits)
    return class_loss

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

class EncDecGen(EncDec):
  """
  EncDec for text generation
  """
  def __init__(self, hparams, embedding, emb_dim):
    # Must train output_layer and recycle later for inference
    vocab_size = embedding.shape[0]
    output_layer = tf.contrib.keras.layers.Dense(vocab_size, use_bias=False)
    super().__init__(hparams, embedding, emb_dim, output_layer=output_layer)

    # Sequence outputs over vocab, training
    self.seq_logits = self.decoded_outputs.rnn_output

    # Generator loss per sample
    # self.loss = self.sequence_loss(\
                        # self.seq_logits, self.dec_targets, self.dec_weight_mask)
    self.loss = self.sequence_loss(\
                        self.seq_logits, self.dec_targets, self.dec_input_len)
    self.cost = tf.reduce_mean(self.loss) # average across batch

    # Loss ###################
    self.optimizer = self.get_optimizer(hparams.l_rate).minimize(\
                                      self.cost, global_step=self.global_step)

    # Generated text ###################
    # Sequence outputs over vocab, inferred
    self.infer_outputs, self.infer_final_state, self.infer_final_seq_len=\
        self.decoder_infer(self.batch_size, self.attn_cell, self.initial_state, output_layer)
    self.sample_id = self.infer_outputs.sample_id

  def sequence_loss(self, logits, targets, seq_len):
    """ Loss on sequence, given logits and one-hot targets
    Default loss below is softmax cross ent on logits
    Arguments:
      logits : logits over predictions, [batch, seq_len, num_decoder_symb]
      targets : the class id, shape is [batch_size, seq_len], dtype int
      weigth_mask : valid logits should have weight "1" and padding "0",
        [batch_size, seq_len] of dtype float
    """
    mask = tf.sequence_mask(seq_len, dtype=tf.float32)

    # We need to delete zeroed elements in targets, beyond max sequence
    max_seq = tf.reduce_max(seq_len)
    max_seq = tf.to_int32(max_seq)
    # Slice time dimension to max_seq
    logits = tf.slice(logits, [0, 0, 0], [-1, max_seq, -1])
    targets = tf.slice(targets, [0, 0], [-1, max_seq])
    # weight_mask = tf.slice(weight_mask, [0,0], [-1, max_seq])

    loss = tf.contrib.seq2seq.sequence_loss(logits, targets, mask,
            average_across_batch=False)
    return loss

  def decoder_infer(self, batch_size, attn_cell, initial_state, output_layer):
    """
    Args:
      attn_cell: cell wrapped with attention
      initial_state: initial_state for decoder
      output_layer: Trained dense layer to project output units to vocab

    Returns:
      see decoder_train() above
    """
    # Greedy decoder
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding=self.embedding_tensor,
        start_tokens=tf.tile([hparams.start_token], [batch_size]),
        end_token=hparams.end_token)

    # Decoder setup. This decoder takes inputs and states and feeds it to the
    # RNN cell at every timestep
    decoder = tf.contrib.seq2seq.BasicDecoder(
          cell = attn_cell,
          helper = helper, # A Helper instance
          initial_state = initial_state, # initial state of decoder
          output_layer = output_layer) # instance of tf.layers.Layer, like Dense

    # Perform dynamic decoding with decoder object
    # If impute_fnished=True ensures finished states are copied through,
    # corresponding outputs are zeroed out. For proper backprop
    # Maximum iterations: should be fixed for training, can be none for infer
    outputs, final_state, final_sequence_lengths= \
            tf.contrib.seq2seq.dynamic_decode(\
              decoder=decoder,
              impute_finished=True,
              maximum_iterations=hparams.max_seq_len) # if None, decode till stop token
    return outputs, final_state, final_sequence_lengths

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


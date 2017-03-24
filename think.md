Think about these

## Decoders
tf.nn.dynamic_rnn(
tf.nn.bidirectional_dynamic_rnn(
tf.contrib.rnn.stack_bidirectional_dynamic_rnn(

## From TensorFlow dev conf
[watch](https://www.youtube.com/watch?v=RIR_-Xlbp7s&list=PLOU2XLYxmsIKGc_NBoIhTn2Qhraji53cv&index=15)
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

## Train batching
batching with dynamic padding:
tf.train.batch(... dynamic_pad=True)

or have similar length sequences grouped together:
tf.contrib.training.bucket_by_sequence_length(... dynamic_pad=True)

We can automatically trunc sequences for BPTT with a state saver
tf.contrib.training.batch_sequences_with_states(...)

## trainer? Pipeline?
helper = TrainingHelper(decoder_inputs, sequence_length)
'''

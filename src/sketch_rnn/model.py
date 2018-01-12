import tensorflow as tf
from tensorflow.contrib import rnn

from .config import model


class SketchRNN():

    def __init__(self, x, y, seq_len, n_class, cell_hidden=[128, ], n_hidden=128):

        assert model == 1

        self.n_class = n_class
        self.n_hidden = n_hidden
        self.cell_hidden = cell_hidden

        self.pred = self.network(x, seq_len)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.pred, labels=tf.one_hot(y, n_class)))


    def network(self, x, seq_len, reuse=False):
        with tf.variable_scope("sketch_rnn") as scope:
            if reuse:
                scope.reuse_variables()
            init = tf.truncated_normal_initializer(stddev=0.01)
            weights = tf.get_variable("fc_w", shape=[self.n_hidden, self.n_class],
                                      dtype=tf.float32, initializer=init)
            bias = tf.get_variable("fc_b", shape=[self.n_class], dtype=tf.float32, initializer=init)

            batch_size = tf.shape(x)[0]
            max_seq_len = tf.shape(x)[1]

            pred = self._dynamic_rnn(x, weights, bias, seq_len, batch_size, max_seq_len)

            return pred


    def _dynamic_rnn(self, x, weights, bias, seq_len, batch_size, max_seq_len):

        cell = rnn.MultiRNNCell([rnn.GRUCell(cell_hidden) for cell_hidden in self.cell_hidden])
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(
            cell,
            inputs=x,
            initial_state=init_state,
            sequence_length=seq_len
        )

        index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)

        outputs = tf.reshape(outputs, [-1, self.n_hidden])
        outputs = tf.gather(outputs, index)
        return tf.matmul(outputs, weights) + bias

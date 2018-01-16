import tensorflow as tf
from tensorflow.contrib import rnn

from .config import model


class SketchRNN():

    def __init__(self, x, y, seq_len, n_class, cell_hidden=[128, ], avg_output=False):

        if not avg_output:
            assert model == 1
        else:
            assert model == 2

        self.n_class = n_class
        self.cell_hidden = cell_hidden
        self.avg_output = avg_output

        self.pred = self.network(x, seq_len)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.pred, labels=tf.one_hot(y, n_class)))


    def network(self, x, seq_len, reuse=False):
        with tf.variable_scope("sketch_rnn") as scope:
            if reuse:
                scope.reuse_variables()
            init = tf.truncated_normal_initializer(stddev=0.01)
            weights = tf.get_variable("fc_w", shape=[self.cell_hidden[-1], self.n_class],
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

        if not self.avg_output:
            index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)
            outputs = tf.reshape(outputs, [-1, self.cell_hidden[-1]])
            outputs = tf.gather(outputs, index)
        else:
            outputs = tf.reduce_mean(outputs, axis=1)

        return tf.matmul(outputs, weights) + bias



class SketchBiRNN():

    def __init__(self, x, y, seq_len, n_class, cell_hidden=[128, ], avg_output=False):

        if not avg_output:
            assert model == 3
        else:
            assert model == 4

        self.n_class = n_class
        self.cell_hidden = cell_hidden
        self.avg_output = avg_output

        self.pred = self.network(x, seq_len)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.pred, labels=tf.one_hot(y, n_class)))

    def network(self, x, seq_len, reuse=False):
        with tf.variable_scope("sketch_birnn") as scope:
            if reuse:
                scope.reuse_variables()
            init = tf.truncated_normal_initializer(stddev=0.01)
            weights = tf.get_variable("fc_w", shape=[self.cell_hidden[-1], self.n_class],
                                      dtype=tf.float32, initializer=init)
            bias = tf.get_variable("fc_b", shape=[self.n_class], dtype=tf.float32, initializer=init)

            batch_size = tf.shape(x)[0]
            max_seq_len = tf.shape(x)[1]

            pred = self._dynamic_birnn(x, weights, bias, seq_len, batch_size, max_seq_len)

            return pred

    def _dynamic_birnn(self, x, weights, bias, seq_len, batch_size, max_seq_len):

        cell_fw = rnn.MultiRNNCell([rnn.GRUCell(cell_hidden) for cell_hidden in self.cell_hidden])
        cell_bw = rnn.MultiRNNCell([rnn.GRUCell(cell_hidden) for cell_hidden in self.cell_hidden])
        init_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
        init_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=x,
            initial_state_fw=init_state_fw,
            initial_state_bw=init_state_bw,
            sequence_length=seq_len
        )

        if not self.avg_output:
            index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)
            outputs = tf.reshape(outputs, [-1, self.cell_hidden[-1]])
            outputs = tf.gather(outputs, index)
        else:
            outputs = tf.reduce_mean(outputs, axis=1)

        return tf.matmul(outputs, weights) + bias




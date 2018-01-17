import tensorflow as tf
from tensorflow.contrib import rnn

from .config import model


class SketchRNN():

    def __init__(self, n_class, cell_hidden=[128, ], avg_output=False):

        if not avg_output:
            assert model == 1
        else:
            assert model == 2

        self.n_class = n_class
        self.avg_output = avg_output
        self.cell_hidden = cell_hidden
        self.fc_hidden = 200


    def train(self, x, y, seq_len):
        pred = self._network(x, seq_len)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=tf.one_hot(y, self.n_class)))
        return pred, loss


    def _network(self, x, seq_len, reuse=False):
        with tf.variable_scope("sketch_rnn") as scope:
            if reuse:
                scope.reuse_variables()
            batch_size = tf.shape(x)[0]
            max_seq_len = tf.shape(x)[1]

            pred = self._dynamic_rnn(x, seq_len, batch_size, max_seq_len)

            return pred

    def inference(self, x, seq_len, reuse=False):
        pred = self._network(x, seq_len, reuse)
        pred = tf.nn.softmax(pred)
        return pred


    def _dynamic_rnn(self, x, seq_len, batch_size, max_seq_len):

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
            # outputs = tf.Print(outputs, [tf.shape(outputs)], message="output shape")
            outputs = tf.reduce_sum(outputs, axis=1)
            outputs = tf.divide(outputs, tf.cast(seq_len[:, None], tf.float32))
            # outputs = tf.Print(outputs, [tf.shape(outputs)], message="output shape")


        fc1 = tf.layers.dense(outputs, self.fc_hidden, name="fc1")
        fc2 = tf.layers.dense(fc1, self.n_class, name="fc2")

        return fc2



class SketchBiRNN():

    def __init__(self, n_class, cell_hidden=[128, ], avg_output=False):

        if not avg_output:
            assert model == 3
        else:
            assert model == 4

        self.n_class = n_class
        self.cell_hidden = cell_hidden
        self.avg_output = avg_output
        self.fc_hidden = 200


    def _network(self, x, seq_len, reuse=False):
        with tf.variable_scope("sketch_birnn") as scope:
            if reuse:
                scope.reuse_variables()

            batch_size = tf.shape(x)[0]
            max_seq_len = tf.shape(x)[1]

            pred = self._dynamic_birnn(x, seq_len, batch_size, max_seq_len)

            return pred

    def train(self, x, y, seq_len):
        pred = self._network(x, seq_len)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=pred, labels=tf.one_hot(y, self.n_class)))
        return pred, cost


    def inference(self, x, seq_len, reuse=False):
        pred = self._network(x, seq_len, reuse)
        pred = tf.nn.softmax(pred)
        return pred

    def _dynamic_birnn(self, x, seq_len, batch_size, max_seq_len):

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

        outputs = tf.concat(outputs, 2)

        if not self.avg_output:
            index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)
            outputs = tf.reshape(outputs, [-1, self.cell_hidden[-1] * 2])
            outputs = tf.gather(outputs, index)
        else:
            outputs = tf.reduce_sum(outputs, axis=1)
            outputs = tf.divide(outputs, tf.cast(seq_len[:, None], tf.float32))

        fc1 = tf.layers.dense(outputs, self.fc_hidden, name="fc1")
        fc2 = tf.layers.dense(fc1, self.n_class, name="fc2")

        return fc2




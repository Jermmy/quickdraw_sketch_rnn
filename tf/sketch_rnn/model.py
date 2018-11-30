import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, MultiRNNCell

from .config import model


class SketchRNN:

    def __init__(self, n_class, cell_hidden=[128, ], avg_output=False):

        if not avg_output:
            assert model == 1
        else:
            assert model == 2

        self.n_class = n_class
        self.avg_output = avg_output
        self.cell_hidden = cell_hidden

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

        cell = MultiRNNCell([GRUCell(cell_hidden) for cell_hidden in self.cell_hidden])
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

        fc = tf.layers.dense(outputs, 1000)
        fc = tf.nn.leaky_relu(fc, 0.2)
        fc = tf.layers.dense(fc, self.n_class)

        return fc


class SketchBiRNN:

    def __init__(self, n_class, cell_hidden=[128, ], avg_output=False):

        if not avg_output:
            assert model == 3
        else:
            assert model == 4

        self.n_class = n_class
        self.cell_hidden = cell_hidden
        self.avg_output = avg_output

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

        cell_fw = MultiRNNCell([GRUCell(cell_hidden) for cell_hidden in self.cell_hidden])
        cell_bw = MultiRNNCell([GRUCell(cell_hidden) for cell_hidden in self.cell_hidden])
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

        # outputs = tf.concat(outputs, 2)
        #
        # if not self.avg_output:
        #     index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)
        #     outputs = tf.reshape(outputs, [-1, self.cell_hidden[-1] * 2])
        #     outputs = tf.gather(outputs, index)
        # else:
        #     outputs = tf.reduce_sum(outputs, axis=1)
        #     outputs = tf.divide(outputs, tf.cast(seq_len[:, None], tf.float32))

        outputs = (outputs[0] + outputs[1]) / 2

        if not self.avg_output:
            index = tf.range(0, batch_size) * max_seq_len + (seq_len - 1)
            outputs = tf.reshape(outputs, [-1, self.cell_hidden[-1]])
            outputs = tf.gather(outputs, index)
        else:
            outputs = tf.reduce_sum(outputs, axis=1)
            outputs = tf.divide(outputs, tf.cast(seq_len[:, None], tf.float32))

        fc = tf.layers.dense(outputs, 1000)
        fc = tf.nn.leaky_relu(fc, 0.2)
        fc = tf.layers.dense(fc, self.n_class)

        return fc


class SketchConvRNN:

    def __init__(self, n_class, cell_hidden=[128, ]):

        assert model == 5

        self.n_class = n_class
        self.cell_hidden = cell_hidden

    def train(self, x, y, seq_len):
        pred = self._network(x, seq_len, reuse=False, is_training=True)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=pred, labels=y))
        return pred, cost

    def inference(self, x, seq_len, reuse=False):
        pred = self._network(x, seq_len, reuse, is_training=False)
        pred = tf.nn.softmax(pred)
        return pred

    def _network(self, x, seq_len, reuse=False, is_training=False):
        with tf.variable_scope("sketchconvrnn") as scope:
            if reuse:
                scope.reuse_variables()

            conv_input = x

            # conv_input = tf.layers.batch_normalization(conv_input, training=is_training, name="bn1")

            # The 1 stride make sure the seq_len won't change when fedding into GRU
            # c1 = tf.layers.dropout(conv_input, rate=0.3, training=is_training)
            c1 = tf.layers.conv1d(conv_input, filters=48, kernel_size=5,
                                  padding='SAME', strides=1, name="conv1")
            c1 = tf.layers.batch_normalization(c1, training=is_training, name="bn1")
            c1 = tf.nn.tanh(c1)
            c2 = tf.layers.dropout(c1, rate=0.3, training=is_training)
            c2 = tf.layers.conv1d(c2, filters=64, kernel_size=5,
                                  padding='SAME', strides=1, name="conv2")
            c2 = tf.layers.batch_normalization(c2, training=is_training, name="bn2")
            c2 = tf.nn.tanh(c2)
            # c3 = tf.layers.dropout(c2, rate=0.3, training=is_training)
            # c3 = tf.layers.conv1d(c3, filters=96, kernel_size=3,
            #                       padding='SAME', strides=1, name="conv3")

            batch_size = tf.shape(x)[0]
            max_seq_len = tf.shape(x)[1]

            pred = self._dynamic_birnn(c2, seq_len, batch_size, max_seq_len)

            return pred

    def _dynamic_birnn(self, x, seq_len, batch_size, max_seq_len):

        cell_fw = MultiRNNCell([DropoutWrapper(GRUCell(cell_hidden))
                                    for cell_hidden in self.cell_hidden])
        cell_bw = MultiRNNCell([DropoutWrapper(GRUCell(cell_hidden)) for cell_hidden in self.cell_hidden])

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

        outputs = (outputs[0] + outputs[1]) / 2

        outputs = tf.reduce_sum(outputs, axis=1)
        outputs = tf.divide(outputs, tf.cast(seq_len[:, None], tf.float32))

        fc = tf.layers.dense(outputs, 1000)
        fc = tf.nn.leaky_relu(fc, 0.2)
        fc = tf.layers.dense(fc, self.n_class)

        return fc


def online_model(n_class, cell_hidden=[128, ]):
    if model == 1:
        return SketchRNN(n_class, cell_hidden)
    elif model == 2:
        return SketchRNN(n_class, cell_hidden, avg_output=True)
    elif model == 3:
        return SketchBiRNN(n_class, cell_hidden)
    elif model == 4:
        return SketchBiRNN(n_class, cell_hidden, avg_output=True)
    elif model == 5:
        return SketchConvRNN(n_class, cell_hidden)
    else:
        return None

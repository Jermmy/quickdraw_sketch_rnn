from ..sketch_rnn.model import online_model
from ..sketch_rnn.utils import get_sketch_labels
from ..sketch_rnn.config import model_file

import tensorflow as tf
import numpy as np

class SearchEngine():

    def __init__(self):
        dictionary, reverse_dict = get_sketch_labels()
        n_class = len(dictionary)

        self.sketchrnn = online_model(n_class=n_class, cell_hidden=[500,])
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess=self.sess, save_path=model_file)

    def search(self, sketch):
        pass

    def _preprocess(self, sketch):
        sketch = np.array(sketch)



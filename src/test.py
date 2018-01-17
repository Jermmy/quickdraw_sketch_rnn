from sketch_rnn.config import quick_draw_dir, sketch_label_file, preprocessed_data_dir
from sketch_rnn.utils import load_data_files
import numpy as np
import tensorflow as tf

out = np.array([[[1,2,3],[2,3,4], [0,0,0]], [[0,1,3],[0,2,1],[0,0,1]]])
seq = np.array([2,3])

a = tf.Variable(out)
a = tf.reduce_sum(a, 1)
b = tf.Variable(seq)
c = tf.divide(a, b[:,None])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
c = sess.run(c)
print(c)

# print(np.mean(out, axis=1))
# print(seq)
# print(seq[0:-1])
# print(seq[:, None])
print(np.divide(np.sum(out, 1), seq[:, None]))
print("sum")
print(np.sum(out, 1))
print(np.sum(out, 1) / seq[0:-1])
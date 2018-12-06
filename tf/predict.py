from sketch_rnn.model import online_model
from sketch_rnn.utils import get_sketch_labels, build_line
from sketch_rnn.config import model_file, log_dir
import tensorflow as tf
import numpy as np
import json


def predict():

    dictionary, reverse_dict = get_sketch_labels()
    n_class = len(dictionary)

    sketch_data = []
    key_id = []
    with open('data/test_simplified.csv', 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i].strip()
            drawing = line.split('\"')[1]
            drawing = build_line(json.loads(drawing))
            sketch_data += [drawing]
            id = line.split(',')[0]
            key_id += [id]

    sketchrnn = online_model(n_class, cell_hidden=[256, 512])
    sketch = tf.placeholder(shape=(None, None, 3), dtype=tf.float32)
    sketch_len = tf.placeholder(shape=(None,), dtype=tf.int32)

    pred = sketchrnn.inference(sketch, sketch_len)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()

        saver.restore(sess, model_file)
        print('Loading model %s' % model_file)

        submission = open('data/submission.csv', 'w')
        submission.write('key_id,word\n')

        step = 100

        for i in range(0, len(sketch_data), step):
            print(i)

            sketch_batch = sketch_data[i:min(i+step, len(sketch_data))]
            sketch_len_batch = []
            max_seq = 0
            for s in sketch_batch:
                if s.shape[0] > max_seq:
                    max_seq = s.shape[0]
            input_sketch = np.zeros(shape=(len(sketch_batch), max_seq, 3), dtype=np.float32)
            for k, s in enumerate(sketch_batch):
                input_sketch[k, :s.shape[0], :] = s
                sketch_len_batch += [s.shape[0]]
            sketch_len_batch = np.array(sketch_len_batch)

            outputs = sess.run(pred, feed_dict={sketch: input_sketch,
                                               sketch_len: sketch_len_batch})

            index = i
            for output in outputs:
                output = np.argsort(output)[::-1][0:3]

                submission.write(key_id[index] + ',')
                for j, o in enumerate(output, 1):
                    o = reverse_dict[o].replace(' ', '_')
                    if j < 3:
                        submission.write(o + ' ')
                    else:
                        submission.write(o + '\n')
                index += 1

        submission.close()


if __name__ == '__main__':
    predict()
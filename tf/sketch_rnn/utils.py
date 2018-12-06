from .config import quick_draw_dir, sketch_label_file, preprocessed_data_dir, \
    train_data_size, test_data_size
import os
from os.path import join, isdir
import json
import re
from sklearn.utils import shuffle
from math import ceil
import numpy as np
from tqdm import tqdm
import tensorflow as tf

try:
    from tensorflow.contrib import data
except ImportError:
    from tensorflow import data


def generate_label_file():
    data_files = load_data_files()
    sketches = []
    for file in data_files:
        sketches.append(file.split('/')[-1].split(".")[0])

    with open(sketch_label_file, 'w') as f:
        for label, s in enumerate(sketches):
            f.write(s + "," + str(label) + "\n")


def get_sketch_labels():
    if not os.path.exists(sketch_label_file):
        generate_label_file()

    dictionary, reverse_dictionary = {}, {}
    with open(sketch_label_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(",")
            dictionary[line[0]] = int(line[1])
            reverse_dictionary[int(line[1])] = line[0]

    return dictionary, reverse_dictionary


def load_data_files():
    data_files = [join(quick_draw_dir, d) for d in os.listdir(quick_draw_dir) if d.endswith(".csv")]

    return data_files

def build_line(drawing):

    sketch_lines = []
    for stroke in drawing:
        X = stroke[0]
        Y = stroke[1]

        for (x, y) in zip(X, Y):
            sketch_lines.append([x, y, 0])

        sketch_lines[-1][2] = 1  # end of stroke

    sketch_lines = np.array(sketch_lines, dtype=np.float32)

    lower = np.min(sketch_lines[:, 0:2], axis=0)
    upper = np.max(sketch_lines[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    sketch_lines[:, 0:2] = (sketch_lines[:, 0:2] - lower) / scale

    sketch_lines[1:, 0:2] -= sketch_lines[0:-1, 0:2]
    # sketch_lines[-1, 3] = 1  # end of drawing
    # sketch_lines[0] = [0, 0, 0, 0]   # start at origin

    return sketch_lines[1:]


def preprocess(train_data_size=train_data_size, test_data_size=test_data_size):

    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_line(lines, writer):
        for line in lines:
            if line != "":
                line = line.strip()
                lines = line.split(',')
                if lines[-3] == "True":
                    drawing = line.split('\"')[1]
                    drawing = json.loads(drawing)
                    sketch_data = build_line(drawing)
                    sketch_len = sketch_data.shape[0]
                    sketch_label = dictionary[lines[-1]]

                    feature = {
                        'sketch': bytes_feature(tf.compat.as_bytes(sketch_data.flatten().tostring())),
                        'label': int64_feature(sketch_label),
                        'len': int64_feature(sketch_len)
                    }

                    example = tf.train.Example(features=tf.train.Features(feature=feature))

                    writer.write(example.SerializeToString())

    # def write_line(lines, writer):
    #     for line in lines:
    #         line = json.loads(line)
    #         if line["recognized"]:
    #             sketch_data = build_line(line["drawing"])
    #             sketch_len = sketch_data.shape[0]
    #             sketch_label = dictionary[line["word"]]
    #
    #             feature = {
    #                 'sketch': bytes_feature(tf.compat.as_bytes(sketch_data.flatten().tostring())),
    #                 'label': int64_feature(sketch_label),
    #                 'len': int64_feature(sketch_len)
    #             }
    #
    #             example = tf.train.Example(features=tf.train.Features(feature=feature))
    #
    #             writer.write(example.SerializeToString())

    print("======== preprocess dataset =========")

    dictionary, reverse_dict = get_sketch_labels()

    data_files = load_data_files()

    train_data_size = train_data_size // len(dictionary)
    test_data_size = test_data_size // len(dictionary)

    print(train_data_size)
    print(test_data_size)

    single_file_size = 10000

    for i in range(ceil(train_data_size / single_file_size)):
        fs = [open(file) for file in data_files]

        lines = []
        for f in tqdm(fs):
            read_lines = f.readlines()
            read_lines = read_lines[1:]
            lines.extend(read_lines[i * single_file_size: min(train_data_size, (i + 1) * single_file_size)])
        lines = shuffle(lines)

        print("write file %s" % ("train" + str(i) + ".tfrecords"))
        writer = tf.python_io.TFRecordWriter(join(preprocessed_data_dir,
                                                  "train" + str(i) + ".tfrecords"))
        write_line(lines, writer)

        writer.close()

        for f in fs:
            f.close()

    '''write test file'''
    fs = [open(file) for file in data_files]

    lines = []
    for f in fs:
        lines.extend(
            f.readlines()[train_data_size: train_data_size + test_data_size])

    print("write file %s" % ("test.tfrecords"))
    writer = tf.python_io.TFRecordWriter(join(preprocessed_data_dir,
                                              "test.tfrecords"))
    write_line(lines, writer)
    writer.close()

    for f in fs:
        f.close()

    print("======== preprocess done ========")


class SketchLoader:

    def __init__(self, batch_size=50, epoch=5):
        self.batch_size = batch_size
        self.epoch = epoch

        self.feature_size = 3

        self.train_dataset, self.test_dataset = self._load_dataset()

    def _load_dataset(self):

        def parse_record(example):
            feature = {
                'sketch': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'len': tf.FixedLenFeature([], tf.int64)
            }

            parsed_feature = tf.parse_single_example(example, feature)
            sketch = tf.decode_raw(parsed_feature['sketch'], tf.float32)
            label = tf.cast(parsed_feature['label'], tf.int32)
            sketch_len = tf.cast(parsed_feature['len'], tf.int32)
            sketch = tf.reshape(sketch, [-1, self.feature_size])
            return sketch, label, sketch_len

        train_files = [preprocessed_data_dir + f for f in os.listdir(preprocessed_data_dir)
                       if re.match(r'train', f) != None]
        test_files = [preprocessed_data_dir + f for f in os.listdir(preprocessed_data_dir)
                      if re.match(r'test', f) != None]

        # train_dataset = tf.data.TFRecordDataset(train_files).map(parse_record).\
        #     apply(data.batch_and_drop_remainder(self.batch_size)).repeat(self.epoch).shuffle(10000)
        # train_dataset = tf.data.TFRecordDataset(train_files). \
        #     map(parse_record). \
        #     padded_batch(self.batch_size, padded_shapes=([None, 4], [])). \
        #     apply(data.batch_and_drop_remainder(self.batch_size)).\
        #     repeat(self.epoch)

        # reference: https://github.com/tensorflow/tensorflow/issues/13745
        #            https://stackoverflow.com/questions/45955241/how-do-i-create-padded-batches-in-tensorflow-for-tf-train-sequenceexample-data-u?rq=1
        train_dataset = tf.data.TFRecordDataset(train_files). \
            map(parse_record, num_parallel_calls=8). \
            padded_batch(self.batch_size, padded_shapes=([None, self.feature_size], [], [])). \
            repeat(self.epoch)
        test_dataset = tf.data.TFRecordDataset(test_files).map(parse_record). \
            padded_batch(self.batch_size, padded_shapes=([None, self.feature_size], [], [])).repeat(1)

        return train_dataset, test_dataset


if __name__ == '__main__':
    # generate_label_file()

    # preprocess(train_data_size, test_data_size)

    sl = SketchLoader(batch_size=100, epoch=1)

    iterator = sl.train_dataset.make_one_shot_iterator()
    sketch, label, sketch_len = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                e = sess.run([sketch, label, sketch_len])
                print(e[0])

        except tf.errors.OutOfRangeError:
            print("end")

    #


from .config import quick_draw_dir, sketch_label_file, preprocessed_data_dir
import os
from os.path import join, isdir
import json
import re
import numpy as np
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
    data_files = [join(quick_draw_dir, d) for d in os.listdir(quick_draw_dir) if d.endswith(".ndjson")]

    return data_files


class SketchLoader():

    def __init__(self, batch_size=50, epoch=5):
        self.batch_size = batch_size
        self.epoch = epoch

        if not os.path.exists(preprocessed_data_dir):
            os.mkdir(preprocessed_data_dir)
            self._preprocess()

        self.train_dataset, self.valid_dataset, self.test_dataset = self._load_dataset()

    def _preprocess(self, train_data_size=50000, valid_data_size=5000, test_data_size=5000):

        def build_line(drawing):

            sketch_lines = []
            for stroke in drawing:
                X = stroke[0]
                Y = stroke[1]

                for (x, y) in zip(X, Y):
                    sketch_lines.append([x, y, 0, 0])

                sketch_lines[-1][2] = 1   # end of stroke

            sketch_lines = np.array(sketch_lines, dtype=np.float32)

            sketch_lines[1:, 0:2] -= sketch_lines[0:-1, 0:2]
            sketch_lines[-1, 3] = 1  # end of drawing
            # sketch_lines[0] = [0, 0, 0, 0]   # start at origin

            return sketch_lines[1:]

        def int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def float32_feature(value):
            return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

        def bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def write_line(line, writer):
            line = json.loads(line)
            if line["recognized"]:
                # sketch_data.append(build_line(line["drawing"]))
                # sketch_label.append(dictionary[line["word"]])

                sketch_data = build_line(line["drawing"])
                sketch_label = dictionary[line["word"]]

                feature = {
                    'train/sketch': bytes_feature(tf.compat.as_bytes(sketch_data.flatten().tostring())),
                    'train/label': int64_feature(sketch_label)
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                writer.write(example.SerializeToString())


        print("======== preprocess dataset =========")

        dictionary, reverse_dict = get_sketch_labels()

        data_files = load_data_files()

        fs = [open(file) for file in data_files]

        record_file = 1

        writer = tf.python_io.TFRecordWriter(join(preprocessed_data_dir,
                                                  "train" + str(record_file) + ".tfrecords"))

        for i in range(train_data_size):
            for f in fs:
                line = f.readline()
                if len(line) == 0:
                    continue

                write_line(line, writer)

            if (i + 1) % 10000 == 0 and i != train_data_size - 1:
                writer.close()
                record_file += 1
                writer = tf.python_io.TFRecordWriter(join(preprocessed_data_dir,
                                                  "train" + str(record_file) + ".tfrecords"))
                print("write file %s" % ("train" + str(record_file) + ".tfrecords"))

            if i == train_data_size - 1:
                writer.close()

        writer = tf.python_io.TFRecordWriter(join(preprocessed_data_dir,
                                                  "valid.tfrecords"))
        for i in range(valid_data_size):
            for f in fs:
                line = f.readline()
                if len(line) == 0:
                    continue
                write_line(line, writer)
        writer.close()
        print("write file %s" % ("valid.tfrecords"))

        writer = tf.python_io.TFRecordWriter(join(preprocessed_data_dir,
                                                  "test.tfrecords"))
        for i in range(test_data_size):
            for f in fs:
                line = f.readline()
                if len(line) == 0:
                    continue
                write_line(line, writer)
        writer.close()
        print("write file %s" % ("test.tfrecords"))

        for f in fs:
            f.close()

        print("======== preprocess done ========")

    def _load_dataset(self):

        def parse_record(example):
            feature = {
                'train/sketch': tf.FixedLenFeature([], tf.string),
                'train/label': tf.FixedLenFeature([], tf.int64)
            }

            parsed_feature = tf.parse_single_example(example, feature)
            sketch = tf.decode_raw(parsed_feature['train/sketch'], tf.float32)
            label = tf.cast(parsed_feature['train/label'], tf.float32)
            sketch = tf.reshape(sketch, [-1, 4])
            return sketch, label

        train_files = [preprocessed_data_dir + f for f in os.listdir(preprocessed_data_dir)
                       if re.match(r'train', f) != None]
        valid_files = [preprocessed_data_dir + f for f in os.listdir(preprocessed_data_dir)
                      if re.match(r'valid', f) != None]
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
            map(parse_record). \
            padded_batch(self.batch_size, padded_shapes=([None, 4], [])). \
            repeat(self.epoch).shuffle(1000)
        valid_dataset = tf.data.TFRecordDataset(valid_files).map(parse_record)
        test_dataset = tf.data.TFRecordDataset(test_files).map(parse_record)

        return train_dataset, valid_dataset, test_dataset




if __name__ == '__main__':
    # generate_label_file()


    sl = SketchLoader(batch_size=100, epoch=1)

    iterator = sl.train_dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    with tf.Session() as sess:
        e = sess.run(one_element)
        # print(e[0])
        print(e[1])







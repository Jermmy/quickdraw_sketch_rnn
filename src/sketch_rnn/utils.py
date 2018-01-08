from .config import quick_draw_dir, sketch_label_file, preprocessed_data_dir
import os
from os.path import join, isdir
import json
import re
import numpy as np
import pickle
import tensorflow as tf


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

    def __init__(self, batch_size=50):
        self.batch_size = batch_size

        if not os.path.exists(preprocessed_data_dir):
            os.mkdir(preprocessed_data_dir)
            self._preprocess()

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

            if (i + 1) % 10000 == 0:
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




if __name__ == '__main__':
    # generate_label_file()


    sl = SketchLoader()

    # with open(preprocessed_data_dir + "train1.pkl", 'rb') as f:
    #     save = pickle.load(f)
    #
    #     sketch = save["sketch"]
    #
    #     print(len(sketch))
    #
    #     sketch = sketch[0]
    #     print(sketch)
    #     sketch = sketch.flatten()
    #     print(sketch)
    #
    #     sketch = sketch.reshape(-1, 4)
    #     print(sketch)
    #
    #     del save







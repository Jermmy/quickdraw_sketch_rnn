from .config import quick_draw_dir, sketch_label_file
import os
from os.path import join, isdir
import json
import re


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




if __name__ == '__main__':
    # generate_label_file()
    dictionary, reverse_dict = get_sketch_labels()

    data_files = load_data_files()

    data = data_files[0]

    # with open(data, 'r') as f:
    #
    #     for line in f.readlines():
    #         record = json.loads(line)
    #         print(record["word"])
    #         print(record["drawing"])



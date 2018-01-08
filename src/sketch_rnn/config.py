import platform
import os

if "Darwin" in platform.system():
    root_dir = "/Users/xyz/Desktop/sketch_test/SHREC13/sketch_rnn/"


quick_draw_dir = root_dir + "DrawShape13/"

preprocessed_data_dir = quick_draw_dir + "preprocessed_data/"

sketch_label_file = root_dir + "label.csv"

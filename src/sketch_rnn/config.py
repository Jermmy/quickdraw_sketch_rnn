import platform
import os
from os.path import join

if "Darwin" in platform.system():
    root_dir = "/Users/xyz/Desktop/sketch_test/SHREC13/sketch_rnn/"
elif "Linux" in platform.system():
    root_dir = ""


quick_draw_dir = root_dir + "DrawShape13/"

preprocessed_data_dir = quick_draw_dir + "preprocessed_data/"

sketch_label_file = root_dir + "label.csv"

train_data_size=500000
valid_data_size=8000
test_data_size=8000

model = 1  # GRU RNN

if model == 1:
    model_dir = join(root_dir, "model1")
    log_dir = join(root_dir, "model1")

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_file = join(model_dir, "model.ckpt")



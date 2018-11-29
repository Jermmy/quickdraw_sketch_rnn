import platform
import os
from os.path import join

root_dir = "/media/liuwq/data/Dataset/quick draw/"

quick_draw_dir = root_dir + "train_simplified/"

preprocessed_data_dir = "data/"

sketch_label_file = preprocessed_data_dir + "label.csv"

# train_data_size=750000
train_data_size = 340 * 50000
test_data_size = 340 * 1000

# model = 1  # GRU RNN with last output states
# model = 2  # GRU RNN with mean output states
# model = 3  # GRU BiRNN with last output states
# model = 4  # GRU BiRNN with mean output states
model = 5  # GRU BiRNN + CNN

model_dir = join('model', "model" + str(model))
log_dir = join('result', "log" + str(model))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_file = join(model_dir, "model.ckpt")


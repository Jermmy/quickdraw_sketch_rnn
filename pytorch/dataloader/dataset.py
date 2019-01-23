import torch
from torch.utils.data import Dataset

import os
from os.path import join
import numpy as np
import pickle
import shelve
import random
from tqdm import tqdm

from utils.util import process_label_file


def build_line(sketch):
    sketch_lines = []
    for stroke in sketch:
        X = stroke[0]
        Y = stroke[1]

        for (x, y) in zip(X, Y):
            sketch_lines += [[x, y, 0]]
        sketch_lines[-1][2] = 1  # end of stroke
    sketch_lines[-1][2] = 2  # end of sketch

    sketch_lines = np.array(sketch_lines, dtype=np.float32)

    min_point = np.min(sketch_lines[:, 0:2], axis=0)
    max_point = np.max(sketch_lines[:, 0:2], axis=0)
    scale = max_point - min_point
    scale[scale == 0] = 1
    sketch_lines[:, 0:2] = (sketch_lines[:, 0:2] - min_point) / scale

    sketch_lines[1:, 0:2] -= sketch_lines[0:-1, 0:2]
    return sketch_lines[1:]


class TrainDataset(Dataset):

    def __init__(self, sketch_feat_file, shuffle=True):

        self.max_seq_len = 0
        self.train_data = shelve.open(sketch_feat_file, flag='r')
        self.train_ids = self.train_data['key_ids']

        if shuffle:
            random.shuffle(self.train_ids)

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):
        cur_id = self.train_ids[idx]
        label, sketch = self.train_data[cur_id]

        sketch = build_line(sketch)
        seq_len = sketch.shape[0]

        # pad_sketch = np.zeros(shape=(self.max_seq_len, sketch.shape[1]))
        # pad_sketch[0:sketch.shape[0]] = sketch

        sample = {'sketch': sketch, 'label': label, 'seq_len': seq_len}
        return sample

class TestDataset(TrainDataset):

    def __init__(self, sketch_feat_file, shuffle=True):
        super(TestDataset, self).__init__(sketch_feat_file, shuffle)
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


def collate_fn(batch):
    batch.sort(key=lambda x: x['seq_len'], reverse=True)
    sketches = [x['sketch'] for x in batch]  # x['sketch']: [seq_len x feat_size]
    labels = np.array([x['label'] for x in batch])
    seq_lens = np.array([x['seq_len'] for x in batch])
    max_seq_len = seq_lens[0]
    pad_sketch = np.zeros(shape=(len(sketches), max_seq_len, sketches[0].shape[1]))
    for i in range(len(sketches)):
        pad_sketch[i, 0:sketches[i].shape[0], :] = sketches[i]
    pad_sketch = torch.from_numpy(pad_sketch)
    labels = torch.from_numpy(labels)
    seq_lens = torch.from_numpy(seq_lens)
    return {'sketch': pad_sketch, 'label': labels, 'seq_len': seq_lens}


class TrainDataset(Dataset):

    def __init__(self, sketch_feat_file, shuffle=True):

        self.max_seq_len = 0
        self.train_data = shelve.open(sketch_feat_file, flag='r')
        self.train_ids = self.train_data['key_ids']
        print("Loading key ids finish!")

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

    def close(self):
        self.train_data.close()


class TestDataset(TrainDataset):

    def __init__(self, sketch_feat_file, shuffle=True):
        super(TestDataset, self).__init__(sketch_feat_file, shuffle)
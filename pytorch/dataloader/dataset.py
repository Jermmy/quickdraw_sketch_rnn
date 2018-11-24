import torch
from torch.utils.data import Dataset

import os
from os.path import join, exists
import numpy as np
import pickle
import random

MAX_SIZE = 1000


class TrainDataset(Dataset):

    def __init__(self, train_dir, label_file):
        self.train_dir = train_dir
        self.max_seq_len = 0
        self.files = [f for f in os.listdir(train_dir) if f.endswith('pkl')]
        self.label_dict = self.process_label_file(label_file) # name: label
        self.train_data, self.train_label, self.permutation = self._read_files(train_dir, self.files, self.label_dict)

    def process_label_file(self, label_file):
        dictionary = {}
        with open(label_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                dictionary[line[0]] = int(line[1])
        return dictionary

    def _read_files(self, train_dir, files, dict):
        train_data = []
        train_label = []
        for i, file in enumerate(files):
            print('read %s' % file)
            with open(join(train_dir, file), 'rb') as f:
                data = pickle.load(f)
                random.shuffle(data)
                # We load only MAX_SIZE in each iteration to avoid OOM
                sample_data = data[0:MAX_SIZE]
                for sketch in sample_data:
                    sketch = self._build_line(sketch)
                    train_data += [sketch]
                    train_label += [dict[file.split('.')[0]]]
                    if len(sketch) > self.max_seq_len:
                        self.max_seq_len = len(sketch)

                # train_data.extend(sample_data)
                # train_label.extend([dict[file.split('.')[0]]] * MAX_SIZE)

        permutation = np.random.permutation(len(train_data))
        return train_data, train_label, permutation

    def reload_pkl_files(self):
        '''
        Reload dataset after each iteration
        :return:
        '''
        print('reload dataset')
        self.train_data, self.train_label, self.permutation = self._read_files(self.train_dir, self.files, self.label_dict)

    def __len__(self):
        return len(self.permutation)

    def __getitem__(self, idx):
        idx = self.permutation[idx]
        sketch = self.train_data[idx]
        label = self.train_label[idx]
        sample = {'sketch': sketch, 'label': label}
        return sample

    def _build_line(self, data):
        sketch_lines = []
        for stroke in data:
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


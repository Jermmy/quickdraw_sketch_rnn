import torch
from torch.utils.data import Dataset

import os
from os.path import join, exists
import numpy as np
import pickle
import random

MAX_SIZE = 1


class TrainDataset(Dataset):

    def __init__(self, train_dir, label_file):
        self.train_dir = train_dir
        self.files = [f for f in os.listdir(train_dir) if f.endswith('npy')]
        self.label_dict = self.process_label_file(label_file) # name: label
        self.train_data = self._read_files(train_dir, self.files, self.label_dict)

    def process_label_file(self, label_file):
        dictionary = {}
        with open(label_file, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                dictionary[line[0]] = int(line[1])
        return dictionary

    def _read_files(self, train_dir, files, dict):
        train_datas = {}
        for file in files:
            print('read %s' % file)
            with open(join(train_dir, file), 'rb') as f:
                data = pickle.load(f)
                label = dict[file.split('.')[0]]
                np.random.shuffle(data)
                data = data[0:MAX_SIZE]
                # We load only MAX_SIZE in each iteration to avoid OOM
                train_datas[label] = data

        return train_datas

    def reload_npy_files(self):
        '''
        Reload dataset after each iteration
        :return:
        '''
        print('reload dataset')
        self.train_data = self._read_files(self.train_dir, self.files, self.label_dict)

    def __len__(self):
        size = 0
        for k, v in self.train_data.items():
            size += len(v)
        return size

    def __getitem__(self, idx):
        rand_label = random.sample(self.label_dict.values(), k=1)
        sketch = np.random.choice(self.train_data[rand_label], 1)
        sketch = self._build_line(sketch)
        sample = {'sketch': sketch, 'label': rand_label}
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


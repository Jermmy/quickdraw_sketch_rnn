import os
from os.path import join, exists
import numpy as np
import argparse
import json

import torch

from model.sketch_rnn import SketchRNN
from dataloader.dataset import build_line, process_label_file


def predict(config):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    sketch = []
    key_id = []
    with open(config.test_file, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            line = lines[i].strip()
            drawing = line.split('\"')[1]
            drawing = build_line(json.loads(drawing))
            sketch += [drawing]
            id = line.split(',')[0]
            key_id += [id]

    dictionary, reverse_dict = process_label_file(config.label_file)

    sketchrnn = SketchRNN(config.input_size, config.hidden_size, output_size=len(dictionary.keys()),
              n_layers=config.n_layers, device=device).to(device)

    if config.load_model:
        sketchrnn.load_state_dict(torch.load(config.load_model))

    submission = open(config.submission_file, 'w')
    submission.write('key_id,word\n')

    sketchrnn.eval()
    for i in range(len(sketch)):
        s = torch.from_numpy(np.array([sketch[i]])).to(device)
        seq_len = torch.Tensor([s.shape[1]]).int().to(device)
        hidden = sketchrnn.initHidden(s.shape[0]).to(device)

        output, hidden, new_idx = sketchrnn(s, hidden, seq_len)
        output = np.argsort(output.detach().cpu().numpy()[0])[::-1][0:3]

        submission.write(key_id[i] + ',')
        for j, o in enumerate(output, 1):
            o = reverse_dict[o].replace(' ', '_')
            if j < 3:
                submission.write(o + ' ')
            else:
                submission.write(o + '\n')

    submission.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/test_simplified.csv')
    parser.add_argument('--submission_file', type=str, default='data/submission.csv')
    parser.add_argument('--label_file', type=str, default='/media/liuwq/data/Dataset/quick draw/label.csv')
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--load_model', type=str, default=None)

    config = parser.parse_args()
    predict(config)



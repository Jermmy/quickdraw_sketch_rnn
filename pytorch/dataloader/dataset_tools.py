import os
from os.path import join, exists
import sys
import argparse
import numpy as np
import json
import pickle

# ----------------------------------------------------------
def generate_label_dict(quickdraw_dir, label_file):
    names = [f.split('.')[0] for f in os.listdir(quickdraw_dir) if f.endswith('csv')]
    dict = {}
    for label, name in enumerate(names):
        dict[name] = label
    with open(label_file, 'w') as f:
        for name, label in dict.items():
            f.write(name + ',' + str(label) + '\n')

# ----------------------------------------------------------
def split_dataset(quickdraw_dir, train_dir, test_dir):

    def process_line(line):
        if line != "":
            line = line.strip()
            lines = line.split(',')
            if lines[-3] == "True":
                drawing = line.split('\"')[1]
                drawing = json.loads(drawing)
                return drawing
            else:
                return None
        else:
            return None

    if not exists(train_dir):
        os.makedirs(train_dir)
    if not exists(test_dir):
        os.makedirs(test_dir)

    data_files = [f for f in os.listdir(quickdraw_dir) if f.endswith('csv')]

    train_max_size = 0
    train_min_size = 1000000

    for file in data_files:
        with open(join(quickdraw_dir, file), 'r') as f:
            lines = f.readlines()
            train_data = []
            test_data = []
            for i in range(1, int(len(lines) * 0.8)):
                line = process_line(lines[i])
                if line:
                    train_data += [line]

            for i in range(int(len(lines) * 0.8), len(lines)):
                line = process_line(lines[i])
                if line:
                    test_data += [line]

            with open(join(train_dir, file.split('.')[0] + '.pkl'), 'wb') as f:
                pickle.dump(train_data, f)

            with open(join(test_dir, file.split('.')[0] + '.pkl'), 'wb') as f:
                pickle.dump(test_data, f)

            if len(train_data) > train_max_size:
                train_max_size = len(train_data)
            if len(train_data) < train_min_size:
                train_min_size = len(train_data)

            print("Process %s, train data: %d, test data: %d" % (file, len(train_data), len(test_data)))

    print("Finish. Max train size: %d, min train size: %d" % (train_max_size, train_min_size))

# ----------------------------------------------------------
def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog=prog
    )

    subparsers = parser.add_subparsers(dest='command')

    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('generate_label_dict', 'Generate label file',
                    'generate_label_dict train_simplified/ labels.csv')
    p.add_argument('quickdraw_dir', help='Directory to read simplified data')
    p.add_argument('label_file', help='Label file to write')

    p = add_command('split_dataset', 'Split the whole dataset into train and test part.',
                    'split_dataset train_simplified/ train/ test/')
    p.add_argument('quickdraw_dir', help='Directory to read simplified data')
    p.add_argument('train_dir', help='Directory to save train simplified data')
    p.add_argument('test_dir', help='Directory to save test simplified data')

    args = parser.parse_args(argv[1:])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

# ------------------------------------------------------------

if __name__ == '__main__':
    execute_cmdline(sys.argv)
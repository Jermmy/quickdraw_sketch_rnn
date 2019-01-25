import os
from os.path import join, exists
import sys
import argparse
import json
import shelve
import random
from tqdm import tqdm

from utils.util import process_label_file

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


def split_dataset(quickdraw_dir, label_file, train_feat_file, test_feat_file):

    def process_line(line):
        if line != "":
            line = line.strip()
            lines = line.split(',')
            if lines[-3] == "True":
                drawing = line.split('\"')[1]
                drawing = json.loads(drawing)
                key_id = lines[-4]
                return drawing
            else:
                return None
        else:
            return None

    label_dict, _ = process_label_file(label_file)  # name: label

    data_files = [f + '.csv' for f in label_dict.keys()]

    train_database = shelve.open(train_feat_file)
    test_database = shelve.open(test_feat_file)

    train_max_size = 50000
    test_max_size = 10000

    train_key_ids = []
    test_key_ids = []

    train_index = 0
    test_index = 0

    for file in tqdm(data_files):

        with open(join(quickdraw_dir, file), 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            count = 0
            label = label_dict[file.split('.')[0]]
            print(label)
            for i in range(1, int(len(lines) * 0.8)):
                data = process_line(lines[i])
                if data:
                    count += 1
                    train_key_ids += [str(train_index)]
                    train_database[str(train_index)] = [label, data]
                    train_index += 1

                if count >= train_max_size:
                    break

            count = 0

            for i in range(int(len(lines) * 0.8), len(lines)):
                data = process_line(lines[i])
                if data:
                    count += 1
                    test_key_ids += [str(test_index)]
                    test_database[str(test_index)] = [label, data]
                    test_index += 1

                if count >= test_max_size:
                    break

            print("Process %s, train data: %d, test data: %d" % (file, len(train_key_ids), len(test_key_ids)))

    train_database['key_ids'] = train_key_ids
    test_database['key_ids'] = test_key_ids

    train_database.close()
    test_database.close()

    print("Finish")

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
                    'split_dataset train_simplified/ label.csv train/ test/ train.dat test.dat')
    p.add_argument('quickdraw_dir', help='Directory to read simplified data')
    p.add_argument('label_file')
    p.add_argument('train_feat_file', help="Train database file")
    p.add_argument('test_feat_file', help="Test database file")

    args = parser.parse_args(argv[1:])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

# ------------------------------------------------------------


if __name__ == '__main__':
    execute_cmdline(sys.argv)

import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import os
from os.path import join, exists

from dataloader.dataset import TrainDataset, TestDataset
from model.sketch_rnn import GRU
from utils.metrics import mapk, apk


def parse_label(label_file):
    dictionary, reverse_dict = {}, {}
    with open(label_file, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            dictionary[line[0]] = int(line[1])
            reverse_dict[int(line[1])] = line[0]
    return dictionary, reverse_dict


def train(config):
    print(config)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    with open(join(config.result_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    writer = SummaryWriter(config.result_path)

    dictionary, _ = parse_label(config.label_file)

    train_dataset = TrainDataset(config.train_dir, config.label_file)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    test_dataset = TestDataset(config.test_dir, config.label_file)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    gru = GRU(config.input_size, config.hidden_size, output_size=len(dictionary.keys()), n_layers=config.n_layers).to(device)

    nllLoss = torch.nn.NLLLoss().to(device)

    if config.load_model:
        gru.load_state_dict(torch.load(config.load_model))

    lr = config.lr

    optim = torch.optim.RMSprop(params=gru.parameters(), lr=lr)

    for epoch in range(1 + config.start_idx, config.epochs + 1):

        # ===================== Train =================================

        for i, data in enumerate(train_loader):
            sketch = data['sketch'].to(device)
            label = data['label'].to(device)
            seq_len = data['seq_len'].to(device)

            optim.zero_grad()

            hidden = gru.initHidden(sketch.shape[0])

            output, hidden, new_idx = gru(sketch, hidden, seq_len)

            label = label[new_idx]

            loss = nllLoss(output, label)

            loss.backward()
            optim.step()

            if i % 10 == 0:
                print('loss: %.4f' % (loss.item()))

            if i % 1000 == 0:
                writer.add_scalar('loss', loss.item(), (epoch - 1) * len(train_loader) + i)

        # ======================= Evaluation =============================
        gru.eval()
        actuals, predicts = [], []
        for data in test_loader:
            sketch = data['sketch'].to(device)
            label = data['label'].to(device)
            seq_len = data['label'].to(device)
            hidden = gru.initHidden(sketch.shape[0])
            output, hidden, new_idx = gru(sketch, hidden, seq_len)
            label = label[new_idx].detach().numpy().reshape(-1, 1).tolist()
            actuals.extend(label)
            predict = np.argsort(output.detach().numpy(), axis=1)[:, ::-1][:,0:3].tolist()
            predicts.extend(predict)

        accuracy = mapk(actuals, predicts, k=3)
        print('accuracy: %.3f' % (accuracy))
        writer.add_scalar('accuracy', accuracy, epoch)
        gru.train()

        if epoch % 2 == 0:
            train_dataset.reload_pkl_files()

        torch.save(gru.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, default='ckpt/gru')
    parser.add_argument('--result_path', type=str, default='result/gru')

    # parser.add_argument('--label_file', type=str, default='/media/liuwq/data/Dataset/quick draw/label.csv')
    # parser.add_argument('--train_dir', type=str, default='/media/liuwq/data/Dataset/quick draw/train')
    # parser.add_argument('--test_dir', type=str, default='/media/liuwq/data/Dataset/quick draw/test')

    parser.add_argument('--label_file', type=str, default='data/label.csv')
    parser.add_argument('--train_dir', type=str, default='data/')
    parser.add_argument('--test_dir', type=str, default='data/')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)

    config = parser.parse_args()
    train(config)
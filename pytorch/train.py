import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
import os
from os.path import join, exists

from tqdm import tqdm

from dataloader.dataset import TrainDataset, TestDataset, collate_fn
from model.sketch_rnn import SketchRNN
from utils.metrics import mapk, apk
from utils.util import process_label_file


def train(config):
    print(config)

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    if not exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not exists(config.result_path):
        os.makedirs(config.result_path)

    with open(join(config.result_path, 'config.txt'), 'w') as f:
        f.write(str(config))

    writer = SummaryWriter(config.result_path)

    dictionary, _ = process_label_file(config.label_file)

    train_dataset = TrainDataset(config.train_sketch_file)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, drop_last=True, collate_fn=collate_fn)

    test_dataset = TestDataset(config.test_sketch_file)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=1, drop_last=True, collate_fn=collate_fn)

    sketchrnn = SketchRNN(config.input_size, config.hidden_size, output_size=len(dictionary.keys()),
                          n_layers=config.n_layers,
                          device=device,
                          avg_out=(True if config.avg_out == 1 else False),
                          bi_rnn=(True if config.bi_rnn == 1 else False),
                          rnn_type=config.rnn_type,
                          use_conv=(True if config.use_conv == 1 else False)).to(device)

    # criterion = torch.nn.NLLLoss().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    if config.load_model:
        sketchrnn.load_state_dict(torch.load(config.load_model))

    lr = config.lr

    optim = torch.optim.RMSprop(params=sketchrnn.parameters(), lr=lr)

    for epoch in range(1 + config.start_idx, config.epochs + 1):

        # ===================== Train =================================

        for i, data in enumerate((train_loader)):
            sketch = data['sketch'].float().to(device)
            label = data['label'].to(device)
            seq_len = data['seq_len'].to(device)

            # print(label)
            # print(seq_len)

            optim.zero_grad()

            # hidden = sketchrnn.initHidden(sketch.shape[0]).to(device)

            output, new_idx = sketchrnn(sketch, seq_len)

            label = label[new_idx]

            loss = criterion(output, label)

            loss.backward()

            torch.nn.utils.clip_grad_norm(sketchrnn.parameters(), config.grad_clip)

            optim.step()

            if i % 2000 == 0:
                predict = np.argsort(output.detach().cpu().numpy(), axis=1)[:, ::-1][:,0:3].tolist()
                actual = [[l] for l in label.detach().cpu().numpy().tolist()]
                print('Epoch: %d/%d | Step: %d/%d | Loss: %.4f | Accuracy: %.4f' %
                      (epoch, config.epochs, i, len(train_loader), loss.item(), mapk(actual, predict)))

            if i % 5000 == 0:
                writer.add_scalar('loss', loss.item(), (epoch - 1) * len(train_loader) + i)

        # ======================= Evaluation =============================
        print('Evaluation')
        sketchrnn.eval()
        actuals, predicts = [], []
        for data in test_loader:
            sketch = data['sketch'].float().to(device)
            label = data['label'].to(device)
            seq_len = data['seq_len'].to(device)
            output, new_idx = sketchrnn(sketch, seq_len)
            label = label[new_idx].detach().cpu().numpy().reshape(-1, 1).tolist()
            actuals.extend(label)
            predict = np.argsort(output.detach().cpu().numpy(), axis=1)[:, ::-1][:,0:3].tolist()
            predicts.extend(predict)

        accuracy = mapk(actuals, predicts, k=3)
        print('accuracy: %.3f' % (accuracy))
        writer.add_scalar('accuracy', accuracy, epoch)
        sketchrnn.train()

        if epoch % 10 == 0:
            lr /= 10
            optim = torch.optim.RMSprop(params=sketchrnn.parameters(), lr=lr)

        torch.save(sketchrnn.state_dict(), join(config.ckpt_path, 'epoch-%d.pkl' % epoch))

    writer.close()

    train_dataset.close()
    test_dataset.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_path', type=str, default='ckpt/gru')
    parser.add_argument('--result_path', type=str, default='result/gru')

    # parser.add_argument('--train_sketch_file', type=str, default='/media/liuwq/data/Dataset/quick draw/train_feat')
    # parser.add_argument('--test_sketch_file', type=str, default='/media/liuwq/data/Dataset/quick draw/test_feat')
    parser.add_argument('--train_sketch_file', type=str, default='data/train_feat')
    parser.add_argument('--test_sketch_file', type=str, default='data/test_feat')
    parser.add_argument('--label_file', type=str, default='/media/liuwq/data/Dataset/quick draw/label.csv')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--input_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=1.)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--rnn_type', type=str, default='gru')
    parser.add_argument('--avg_out', type=int, default=0)
    parser.add_argument('--bi_rnn', type=int, default=0)
    parser.add_argument('--use_conv', type=int, default=0)

    config = parser.parse_args()
    train(config)

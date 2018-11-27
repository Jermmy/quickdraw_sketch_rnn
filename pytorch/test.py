import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import utils as nn_utils
import numpy as np

batch_size = 2
max_length = 3
hidden_size = 2
n_layers = 1

tensor_in = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(2, 3, 1)
tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = np.array([1, 3])  # list of integers holding information about the batch size at each sequence step

# pack it
order_idx = np.argsort(seq_lengths)[::-1]
print(tensor_in)

print(order_idx)
order_tensor = tensor_in[order_idx.tolist()]
order_seq = seq_lengths[order_idx]
print(order_tensor)

pack = nn_utils.rnn.pack_padded_sequence(order_tensor, order_seq, batch_first=True)

print(pack)

# initialize
rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))

# forward
out, _ = rnn(pack, h0)

# unpack
unpacked = nn_utils.rnn.pad_packed_sequence(out)
out, bz = unpacked[0], unpacked[1]
print('111', out, bz)
# seq_len x batch_size x hidden_size --> batch_size x seq_len x hidden_size
out = out.permute((1, 0, 2))
print('222', out)
bz = (bz-1).view(bz.shape[0], 1, -1)
# bz = bz.repeat(2, 2)
print(bz)
bz = bz.repeat(1, 1, 2)
print(bz)
out = torch.gather(out, 1, bz)
print('333', out)



a = torch.Tensor([[[1, 2, 3], [1,2,1]], [[1, 2, 4], [0, 0, 0]]])
seq = torch.Tensor([2, 1])
seq = seq.view((a.shape[0], 1))
b = torch.sum(a, dim=1)
c = torch.div(b, seq)
print(a)
print(b)
print(c)
# # import torch
# # import torch.nn as nn
# # from torch.autograd import Variable
# # from torch.nn import utils as nn_utils
# # import numpy as np
# #
# # batch_size = 2
# # max_length = 3
# # hidden_size = 2
# # n_layers = 1
# #
# # tensor_in = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(2, 3, 1)
# # tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
# # seq_lengths = np.array([1, 3])  # list of integers holding information about the batch size at each sequence step
# #
# # # pack it
# # order_idx = np.argsort(seq_lengths)[::-1]
# # print(tensor_in)
# #
# # print(order_idx)
# # order_tensor = tensor_in[order_idx.tolist()]
# # order_seq = seq_lengths[order_idx]
# # print(order_tensor)
# #
# # pack = nn_utils.rnn.pack_padded_sequence(order_tensor, order_seq, batch_first=True)
# #
# # print(pack)
# #
# # # initialize
# # rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
# # h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
# #
# # # forward
# # out, _ = rnn(pack, h0)
# #
# # # unpack
# # unpacked = nn_utils.rnn.pad_packed_sequence(out)
# # out, bz = unpacked[0], unpacked[1]
# # print('111', out, bz)
# # # seq_len x batch_size x hidden_size --> batch_size x seq_len x hidden_size
# # out = out.permute((1, 0, 2))
# # print('222', out)
# # bz = (bz-1).view(bz.shape[0], 1, -1)
# # # bz = bz.repeat(2, 2)
# # print(bz)
# # bz = bz.repeat(1, 1, 2)
# # print(bz)
# # out = torch.gather(out, 1, bz)
# # print('333', out)
# #
# #
# #
# # a = torch.Tensor([[[1, 2, 3], [1,2,1]], [[1, 2, 4], [0, 0, 0]]])
# # seq = torch.Tensor([2, 1])
# # seq = seq.view((a.shape[0], 1))
# # b = torch.sum(a, dim=1)
# # c = torch.div(b, seq)
# # print(a)
# # print(b)
# # print(c)
#
#
#
# import torch
# from torch import nn
# from torch.autograd import Variable
# from torch.nn import utils as nn_utils
# import numpy as np
#
# batch_size = 2
# max_length = 3
# hidden_size = 2
# n_layers = 1
#
# tensor_in = torch.FloatTensor([[1, 0, 0], [1, 2, 3]]).resize_(2, 3, 1)
# tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
# seq_lengths = np.array([1, 3])  # list of integers holding information about the batch size at each sequence step
#
# # pack it
# order_idx = np.argsort(seq_lengths)[::-1]
# print(tensor_in)
#
# print(order_idx)
# order_tensor = tensor_in[order_idx.tolist()]
# order_seq = seq_lengths[order_idx]
# print(order_tensor)
#
# pack = nn_utils.rnn.pack_padded_sequence(order_tensor, order_seq, batch_first=True)
#
# print(pack)
#
# # initialize
# rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
# h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
#
# # forward
# out, _ = rnn(pack, h0)
#
# print(out)
#
# # unpack
# unpacked = nn_utils.rnn.pad_packed_sequence(out)
# out, bz = unpacked[0], unpacked[1]
# print('111')
# print(out, bz)
# # seq_len x batch_size x hidden_size --> batch_size x seq_len x hidden_size
# out = out.permute((1, 0, 2))
# print('222', out)
# bz = (bz-1).view(bz.shape[0], 1, -1)
# # bz = bz.repeat(2, 2)
# print(bz)
# bz = bz.repeat(1, 1, 2)
# print(bz)
# out = torch.gather(out, 1, bz)
# print('333', out)
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import torchvision
#
# torch.manual_seed(1) # reproducible
#
# #  Hyper Parameters
# EPOCH = 1 # 训练整批数据多少次, 为了节约时间, 只训练一次
# BATCH_SIZE = 64
# TIME_STEP = 28 # rnn 时间步数 / 图片高度
# INPUT_SIZE = 28 # rnn 每步输入值 / 图片每行像素
# LR = 0.01 # learning rate
# DOWNLOAD_MNIST = True # 如果你已经下载好了mnist数据就写上 Fasle
#
# #  Mnist 手写数字
# train_data = torchvision.datasets.MNIST(
#     root='./mnist/', # 保存或者提取位置
#     train=True, # this is training data
#     transform=torchvision.transforms.ToTensor(), # 转换 PIL.Image or numpy.ndarray 成 # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
#     download=DOWNLOAD_MNIST, # 没下载就下载, 下载了就不用再下了
# )
#
# test_data = torchvision.datasets.MNIST(root='./mnist/', train=False) # 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
# train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True) # 为了节约时间, 我们测试时只测试前2000个
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255. # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[:2000]
#
#
# class RNN(nn.Module):
#     def __init__(self):
#         super(RNN, self).__init__()
#         self.rnn = nn.LSTM( # LSTM 效果要比 nn.RNN() 好多了
#             input_size=7, # 图片每行的数据像素点
#             hidden_size=64, # rnn hidden unit
#             num_layers=1, # 有几层 RNN layers
#             batch_first=True, # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
#             )
#         self.out = nn.Linear(64, 10) # 输出层
#
#     def forward(self, x):
#         # x shape (batch, time_step, input_size)
#         # r_out shape (batch, time_step, output_size)
#         # h_n shape (n_layers, batch, hidden_size) LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
#         # h_c shape (n_layers, batch, hidden_size)
#         r_out, (h_n, h_c) = self.rnn(x, None)
#         # None 表示 hidden state 会用全0的 state
#         # 选取最后一个时间点的 r_out 输出
#         # 这里 r_out[:, -1, :] 的值也是 h_n 的值
#         out = self.out(r_out[:, -1, :])
#         return out
#
# rnn = RNN()
# print(rnn)
#
# """ RNN ( (rnn): LSTM(28, 64, batch_first=True) (out): Linear (64 -> 10) ) """
#
# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) # optimize all parameters
# loss_func = nn.CrossEntropyLoss() # the target label is not one-hotted
#
# # training and testing
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader): # gives batch data
#        b_x = Variable(x.view(-1, 112, 7)) # reshape x to (batch, time_step, input_size)
#        b_y = Variable(y) # batch y
#        # print(b_y)
#        output = rnn(b_x) # rnn output
#        loss = loss_func(output, b_y) # cross entropy loss
#        optimizer.zero_grad() # clear gradients for this training step
#        loss.backward() # backpropagation, compute gradients
#        optimizer.step() # apply gradients
#        if step % 100 == 0:
#            print('loss: %.4f' % loss.item())
#
#
# test_output = rnn(test_x[:10].view(-1, 112, 7))
# pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# print(pred_y, 'prediction number')
# print(test_y[:10], 'real number')

a = {'1':1, '2':2, '3':3}
print(a[0])

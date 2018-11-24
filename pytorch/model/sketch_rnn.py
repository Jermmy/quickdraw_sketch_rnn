import torch.nn as nn
import torch


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, birnn=False):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.birnn = birnn
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=birnn)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.logSoftmax = nn.LogSoftmax()

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        output = self.h2o(output)
        output = self.logSoftmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

import torch.nn as nn
import torch


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, bi_rnn=False, avg_out=False):
        super(GRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bi_rnn = bi_rnn
        self.avg_out = avg_out
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=bi_rnn)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.logSoftmax = nn.LogSoftmax()

    def forward(self, x, hidden, seq_len):
        '''
        :param x: batch_size x seq_len x input_size
        :param hidden:
        :param seq_len:
        :return:
        '''
        # batch_size x seq_len --> seq_len x batch_size
        x = x.permute((1, 0, 2))
        # Sort seq_len
        desc_seq_len, order_idx = torch.sort(seq_len, descending=True)
        # Pack
        x = torch.nn.utils.rnn.pack_padded_sequence(x.float(), desc_seq_len, batch_first=False)

        output, hidden = self.gru(x, hidden)

        # Unpack
        unpack_padded = torch.nn.utils.rnn.pad_packed_sequence(output)

        output, bz = unpack_padded[0], unpack_padded[1]

        # seq_len x batch_size x hidden_size --> batch_size x seq_len x hidden_size
        output = output.permute((1, 0, 2))

        if self.avg_out:
            pass
        else:
            # e.g. [4,3,2] --> [[[4]], [[3]], [[2]]] (if output_size = 2) --> [[[4,4]], [[3,3]], [[2,2]]]
            index = (bz - 1).view(bz.shape[0], 1, -1).repeat(1, 1, output.shape[-1])

            # Select the last output of sequence,
            # output shape is batch_size x 1 x hidden_size
            output = torch.gather(output, dim=1, index=index)
            output = output.squeeze(1)

        output = self.h2o(output)
        output = self.logSoftmax(output)
        return output, hidden, order_idx

    def initHidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

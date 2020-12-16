import torch
from torch import nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,n_layers):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=self.n_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        h, c = hidden
        output, hidden = self.lstm(embedded, (h, c))
        return output, hidden

    def initHidden(self, batch_size, device):
        hidden = torch.zeros(2, self.n_layers, batch_size, self.hidden_size, device=device)
        return hidden


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, n_layers=3, max_length=512):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.n_layers = n_layers

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1, n_layers=3, max_length=512):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.encoder = EncoderLSTM(input_size, hidden_size,n_layers)
        self.decoder = DecoderLSTM(hidden_size, output_size, dropout_p,n_layers, max_length)
        self.W = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        elif mode == 'predict':
            return self.predict(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    def fwd(self, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None):
        hidden = self.encoder.initHidden(x.size(1), self.encoder.embedding.weight.device)
        output, hidden = self.encoder(x, hidden)
        h, c = hidden
        out = []
        x = x.transpose(1, 0)
        for i in range(x.size(1)):
            input = x[:, i]
            decoder_output, decoder_hidden = self.decoder(input, (h, c))
            h, c = decoder_hidden
            out.append(decoder_output)
        return torch.cat(out, dim=0)

    def predict(self, tensor, pred_mask, y, get_scores):
        masked_tensor = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.hidden_size)
        scores = self.W(masked_tensor)
        loss = F.cross_entropy(scores, y, reduction='mean')
        return scores, loss

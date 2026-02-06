import torch
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        h0_1 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c0_1 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm1(x, (h0_1, c0_1))

        h0_2 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c0_2 = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm2(out, (h0_2, c0_2))

        out = self.fc(out[:, -1, :])  # Take the last time step's output
        return out
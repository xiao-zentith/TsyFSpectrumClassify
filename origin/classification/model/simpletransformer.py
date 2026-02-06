import torch
from torch import nn


class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(SimpleTransformer, self).__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.flatten(x)
        x = self.embedding(x)
        # x = x.permute(1, 0, 2)  # Change shape to (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)
        # x = x.mean(dim=0)  # Global average pooling
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
"""
LSTM Model Definition
A simple 2-layer LSTM for binary classification of index movements.
"""
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        out, _ = self.lstm(x)
        # Take the last sequence step output
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

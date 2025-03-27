import torch
import torch.nn as nn

class BILSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(BILSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # 修正维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hidden, _) = self.bilstm(x)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(last_hidden)
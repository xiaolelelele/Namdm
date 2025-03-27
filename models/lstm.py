import torch
import torch.nn as nn
class LSTMModel(nn.Module):
    """LSTM模型定义"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
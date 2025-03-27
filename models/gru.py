import torch
import torch.nn as nn
class GRUModel(nn.Module):
    """GRU模型定义"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
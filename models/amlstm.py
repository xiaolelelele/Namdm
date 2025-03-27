import torch
import torch.nn as nn
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.u = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
    
    def forward(self, h):
        # h.shape: [batch_size, seq_len, hidden_size] (batch_first=True)
        batch_size, seq_len, hidden_size = h.shape
        
        # 计算注意力得分 p_t = u * tanh(W * h_t)
        energy = self.tanh(self.w(h))  # [batch_size, seq_len, hidden_size]
        scores = self.u(energy).squeeze(-1)  # [batch_size, seq_len]
        
        # 计算注意力权重 a_t (沿序列维度归一化)
        attn_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]
        
        # 加权求和得到上下文向量 beta_t
        # 使用 einsum 标记 "bsh,bs->bh"
        context = torch.einsum("bsh,bs->bh", h, attn_weights)  # [batch_size, hidden_size]
        
        return context, attn_weights

class AMLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # 修改为 batch_first=True
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x.shape: [batch_size, seq_len, input_size] (batch_first=True)
        gru_out, _ = self.lstm(x)  # gru_out: [batch_size, seq_len, hidden_size]
        
        # 应用注意力机制  # context:[batch_size, hidden_size]
        context, attn_weights = self.attention(gru_out)
        
        # 最终预测
        output = self.fc(context)  # [batch_size, output_size]
        
        return output
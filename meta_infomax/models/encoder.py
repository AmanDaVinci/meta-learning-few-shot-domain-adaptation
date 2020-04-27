import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self,emb_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        self.h0 = self.h0.to(x.device)
        self.c0 = self.c0.to(x.device)
        out, _ = self.lstm(x, (self.h0, self.c0))
        return out
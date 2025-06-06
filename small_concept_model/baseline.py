import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple


class LSTMDataset(Dataset):
    """Dataset class to train autoregressive SCM."""

    def __init__(self, data: torch.Tensor):
        assert data.ndim == 3
        self.seq_len = data.shape[1]
        self.data = data.float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        input_seq = seq[: self.seq_len - 1, :]
        target_seq = seq[self.seq_len - 1, :]
        return input_seq, target_seq

class LSTMConceptModel(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_seq):
        """
        x_seq: (batch_size, seq_len, input_dim)
        Returns: (batch_size, input_dim) as the predicted next embedding
        """
        h = self.proj_in(x_seq)
        out_seq, (h_n, c_n) = self.lstm(h)
        last_hidden = h_n[-1]
        last_hidden = self.norm(last_hidden)
        e_pred = self.out(last_hidden)
        return e_pred

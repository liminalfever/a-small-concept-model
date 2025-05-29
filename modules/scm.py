import torch
import torch.nn as nn

class PreNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(PreNet, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def normalize(self, x):
        return (x - self.scaler_mean) / self.scaler_std

    def forward(self, x):
        x = self.normalize(x)
        return self.linear(x)


class PostNet(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int):
        super(PostNet, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.scaler_mean = 0.0
        self.scaler_std = 1.0

    def denormalize(self, x):
        return x * self.scaler_std + self.scaler_mean

    def forward(self, x):
        x = self.linear(x)
        return self.denormalize(x)

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, ff_dim: int, dropout: float = 0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, hidden_dim))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len]
        for layer in self.layers:
            x = layer(x, x)
        return x

class SimpleLCM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super(SimpleLCM, self).__init__()
        self.prenet = PreNet(input_dim, hidden_dim)
        self.transformer = TransformerDecoder(hidden_dim, num_heads, num_layers, ff_dim, dropout)
        self.postnet = PostNet(hidden_dim, output_dim)

    def forward(self, x):
        x = self.prenet(x)
        x = self.transformer(x)
        x = self.postnet(x)
        return x
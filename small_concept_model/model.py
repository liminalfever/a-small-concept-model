import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: Optional[int] = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        denominator = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class InputProj(nn.Module):
    """Input projection layer: d_embed --> d_model."""

    def __init__(
        self,
        d_embed: int,
        d_model: int,
        scaler_mean: Optional[float] = 0.0,
        scaler_std: Optional[float] = 1.0,
    ):
        super(InputProj, self).__init__()
        self.register_buffer("mean", torch.tensor(scaler_mean))
        self.register_buffer("std", torch.tensor(scaler_std))
        self.linear = nn.Linear(d_embed, d_model)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalize(x)
        return self.linear(x)


class OutputProj(nn.Module):
    """Ouptut projection layer: d_model --> d_embed."""

    def __init__(
        self,
        d_model: int,
        d_embed: int,
        scaler_mean: Optional[float] = 0.0,
        scaler_std: Optional[float] = 1.0,
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(scaler_mean))
        self.register_buffer("std", torch.tensor(scaler_std))
        self.linear = nn.Linear(d_model, d_embed)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.std) + self.mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.denormalize(x)


class Transformer(nn.Module):
    """Transformer encoder with causal masking."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: Optional[int] = 4,
        n_layers: Optional[int] = 3,
        dropout: Optional[float] = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=n_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        bool_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        bool_mask = bool_mask.to(x.device)
        return self.transformer(x, mask=bool_mask)


class SmallConceptModel(nn.Module):
    """Autoregressive transformer-based concept model."""

    def __init__(
        self,
        d_model: int,
        d_embed: int,
        d_ff: int,
        n_heads: Optional[int] = 4,
        n_layers: Optional[int] = 3,
        dropout: Optional[float] = 0.1,
        max_seq_len: Optional[int] = 128,
        scaler_mean: Optional[float] = 0.0,
        scaler_std: Optional[float] = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = InputProj(d_embed, d_model, scaler_mean, scaler_std)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.transformer = Transformer(d_model, d_ff, n_heads, n_layers, dropout)
        self.output_projection = OutputProj(d_model, d_embed, scaler_mean, scaler_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.output_projection(x)

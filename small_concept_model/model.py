import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, D = x.size()
        x = x + self.pe[:T]
        return x


def generate_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    """Generates a standard causal mask for autoregressive learning."""

    mask = torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)
    return mask.to(device)


class Normalization(nn.Module):
    """Normalizes the embedding vectors by dimension."""

    def __init__(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
    ):
        super().__init__()
        assert (
            means.dim() == 1 and stds.dim() == 1 and means.size(0) == stds.size(0)
        ), "means and stds must both be 1D tensors of the same length (d_embed)."

        d_embed = means.size(0)
        self.register_buffer('means', means.view(1, 1, d_embed)) # d_model -> (1, 1, d_embed)
        self.register_buffer('stds', stds.view(1, 1, d_embed))   # d_model -> (1, 1, d_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.means) / self.stds


class Denormalization(nn.Module):
    """Denormalizes the embedding vectors by dimension."""

    def __init__(
        self,
        means: torch.Tensor,
        stds: torch.Tensor,
    ):
        super().__init__()
        assert (
            means.dim() == 1 and stds.dim() == 1 and means.size(0) == stds.size(0)
        ), "means and stds must both be 1D tensors of the same length (d_embed)."

        d_embed = means.size(0)
        self.register_buffer('means', means.view(1, 1, d_embed)) # d_model -> (1, 1, d_embed)
        self.register_buffer('stds', stds.view(1, 1, d_embed))   # d_model -> (1, 1, d_embed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.stds + self.means


class SmallConceptModel(nn.Module):
    """The full Small Concept Model (SCM) for next-embedding prediction."""

    def __init__(
        self,
        d_model: int = 384,
        d_embed: int = 384,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 384 * 4,
        dropout: float = 0.1,
        max_seq_len: int = 64,
        means: torch.Tensor = None,
        stds: torch.Tensor = None
    ):
        super().__init__()
        self.d_model = d_model
        self.normalization = Normalization(means, stds)
        self.input_proj = nn.Linear(d_embed, d_model, bias=True)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.output_proj = nn.Linear(d_model, d_embed, bias=True)
        self.denormalization = Denormalization(means, stds)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = input_seq.shape
        device = input_seq.device
        x = self.normalization(input_seq)
        x = x.permute(1, 0, 2)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        causal_mask = generate_causal_mask(T, device=device)
        encoded = self.transformer_encoder(x, mask=causal_mask)
        output = self.output_proj(encoded)
        output = output.permute(1, 0, 2)
        output = self.denormalization(output)
        return output

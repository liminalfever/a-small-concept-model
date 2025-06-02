import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SmallConceptModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 embed_dim: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 max_seq_len: int = 5000):
        super().__init__()
        self.input_projection = nn.Linear(embed_dim, d_model)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(d_model, embed_dim)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(src)
        x_proj = self.pos_encoder(x) * math.sqrt(self.d_model)
        float_mask = torch.zeros_like(src_mask, dtype=torch.float32)
        float_mask = float_mask.masked_fill(src_mask, float("-1e9"))
        enc_output = self.transformer_encoder(x_proj, mask=float_mask)
        return self.output_projection(enc_output)
    
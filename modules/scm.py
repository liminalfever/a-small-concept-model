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

def generate_causal_mask(seq_len: int, device: torch.device):
    """
    Returns a [seq_len x seq_len] mask where mask[i, j] = 0 if j <= i, else -inf (or a large negative),
    so that position i can only attend to positions [0..i].
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    # mask[i,j] = 1 if j > i, else 0. We want to turn 1’s into -inf for attention.
    # PyTorch’s Transformer expects float masks with -inf in illegal places:
    mask = mask.masked_fill(mask == 1, float("-inf"))
    mask = mask.masked_fill(mask == 0, float(0.0))
    return mask  # shape: [seq_len, seq_len]


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, ff_dim: int, dropout: float = 0.1, max_seq_len: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

    def forward(self, x, tgt_mask=None):
        """
        x: Tensor of shape [B, T, hidden_dim]
        tgt_mask: square mask of shape [T, T] containing 0 for allowed, -inf for masked.
        """
        seq_len = x.size(1)
        # Add positional encoding (broadcasted over batch dimension)
        x = x + self.pos_encoder[:, :seq_len, :]
        # TransformerDecoderLayer in PyTorch expects input shape [T, B, hidden_dim], so we must transpose.
        # Indeed, nn.TransformerDecoderLayer expects (tgt, memory, ...), each of shape [T, B, E].
        # We’re using “decoder‐only” (no external memory), so we feed the same x as both tgt and memory.
        # That forces it to attend “only to past” via tgt_mask.
        x = x.transpose(0, 1)  # now shape [T, B, hidden_dim]
        for layer in self.layers:
            x = layer(tgt=x, memory=x, tgt_mask=tgt_mask)
        x = x.transpose(0, 1)  # back to [B, T, hidden_dim]
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
        # x: [B, 16, input_dim]
        x = self.prenet(x)            # [B, 16, hidden_dim]
        causal_mask = generate_causal_mask(x.size(1), device=x.device)  # [16, 16]
        x = self.transformer(x, tgt_mask=causal_mask)  # [B, 16, hidden_dim]
        x = self.postnet(x)           # [B, 16, output_dim]
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizeAndProject(nn.Module):
    """Dimension-wise input normalization and d_embed --> d_model proejction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mean_tensor: torch.Tensor = None,
        std_tensor: torch.Tensor = None,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        means = mean_tensor or torch.zeros([input_dim])
        stds = std_tensor or torch.ones([input_dim])
        self.register_buffer("means", means.view(1, 1, input_dim))
        self.register_buffer("stds", stds.view(1, 1, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.means) / self.stds
        return self.linear(x)


class DenormalizeAndProject(nn.Module):
    """Dimension-wise output denormalization and d_model --> d_embed proejction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mean_tensor: torch.Tensor = None,
        std_tensor: torch.Tensor = None,
    ):
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        means = mean_tensor or torch.zeros([input_dim])
        stds = std_tensor or torch.ones([input_dim])
        self.register_buffer("means", means.view(1, 1, input_dim))
        self.register_buffer("stds", stds.view(1, 1, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.means + (x * self.stds)


class SmallConceptModel(nn.Module):
    """Transformer-based small concept model (SCM) for next-embedding prediction."""  # (B, T, D)

    def __init__(
        self,
        d_embed: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        max_seq_len: int = 16,
        mean_tensor: torch.Tensor = None,
        std_tensor: torch.Tensor = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.in_proj = NormalizeAndProject(d_embed, d_model, mean_tensor, std_tensor)
        self.out_proj = DenormalizeAndProject(d_embed, d_model, mean_tensor, std_tensor)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(transformer_layer, n_layers)
    
    @staticmethod
    def _get_causal_mask(seq_len: int) -> torch.Tensor:
        """Generates standard causal mask for autoregressive training."""

        return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SCM. Feed input with (B, T, D) shape."""

        B, T, D = x.shape
        device = x.device

        x = self.in_proj(x)  # (B, D, T)
        mask = self._get_causal_mask(self.max_seq_len).to(device)
        x = self.decoder(x, mask)  # (B, D, T)
        return self.out_proj(x)  # (B, D, T)

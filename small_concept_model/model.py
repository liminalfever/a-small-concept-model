import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallConceptModel(nn.Module):
    """
    Decoder-only transformer that predicts the next sentence-level embedding
    given a sequence of preceding sentence-level embeddings.

    Input shape: (batch_size, seq_len, d_embed)
    Output shape: (batch_size, seq_len, d_embed)
    """
    def __init__(self,
                 d_embed: int = 384,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_seq_len: int = 16):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.in_proj = nn.Linear(d_embed, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_seq_len, d_model))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, T, d_embed)
        returns: Tensor of shape (B, d_embed), prediction for next embedding
        """
        x = self.in_proj(x)
        B, T, D = x.shape

        assert T <= self.max_seq_len, f"Sequence length {T} > max_seq_len {self.max_seq_len}"

        x = x + self.pos_emb[:T].unsqueeze(0)  # (T, D) -> (1, T, D) broadcast to (B, T, D)
        x = x.permute(1, 0, 2)  # Transformer expects (T, B, D)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)

        memory = torch.zeros(1, B, D, device=x.device)  # memory: omitted or zeros, but nn.TransformerDecoder requires memory: set to zeros.
        dec_out = self.decoder(tgt=x, memory=memory, tgt_mask=tgt_mask)
        dec_out = dec_out.permute(1, 0, 2)  # (T, B, D) -> (B, T, D)

        return self.out_proj(dec_out)
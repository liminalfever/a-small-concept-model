import torch
import torch.nn as nn
import math
from sentence_transformers import SentenceTransformer
from modules.inverter import Inverter, build_inverter, get_encoder
from modules.utils import generate_causal_mask
from modules import params
from typing import List, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class GenerativeSCM:
    def __init__(
        self,
        model: SmallConceptModel,
        encoder: SentenceTransformer,
        inverter: Inverter,
    ):
        self.model = model
        self.encoder = encoder
        self.inverter = inverter

    @staticmethod
    def _generative_inference(model, initial_sequence, n_future_steps, sigma_noise: float = None):
        model.eval()

        prefix = initial_sequence.clone().unsqueeze(0).to(device)  # (1, k, input_dim)
        generated = prefix.clone()

        with torch.no_grad():
            for step in range(n_future_steps):
                current_len = generated.size(1)
                mask = generate_causal_mask(current_len, device=device)
                out = model(generated, mask)
                next_embed = out[:, -1, :].unsqueeze(1)
                if sigma_noise:
                    noise = torch.randn_like(next_embed) * sigma_noise
                    next_embed = next_embed + noise
                generated = torch.cat([generated, next_embed], dim=1)
        return generated
    
    def generate(self, input_texts: List[str], future_steps: Optional[int] = 5, sigma_noise: Optional[float] = 0.01, max_len: Optional[int] = 30):
        if type(input_texts) != list:
            input_texts = [input_texts]
        encoded_inputs = self.encoder.encode(input_texts, convert_to_tensor=True).to(device)
        generated_seq = self._generative_inference(self.model, encoded_inputs, future_steps, sigma_noise=sigma_noise)
        return [self.inverter.invert(v, max_len) for v in generated_seq.squeeze(0)]

    def generate_stream(self, input_texts: List[str], future_steps: Optional[int] = 5, sigma_noise: Optional[float] = 0.01, max_len: Optional[int] = 30):
        if type(input_texts) != list:
            input_texts = [input_texts]
        encoded_inputs = self.encoder.encode(input_texts, convert_to_tensor=True).to(device)
        generated_seq = self._generative_inference(self.model, encoded_inputs, future_steps, sigma_noise=sigma_noise)
        for v in generated_seq.squeeze(0):
            yield self.inverter.invert(v, max_len)
    

def build_scm():
    """Automatically builds SCM loading checkpoints."""
    scm = SmallConceptModel(**params.scm_configs["model_params"]).to(device)
    encoder = get_encoder("all-MiniLM-L6-v2")
    inverter = build_inverter()

    if params.scm_configs["load_checkpoint"]:
        scm.load_state_dict(
            torch.load(params.scm_configs["load_checkpoint"], map_location=device)
        )
    
    return GenerativeSCM(scm, encoder, inverter)
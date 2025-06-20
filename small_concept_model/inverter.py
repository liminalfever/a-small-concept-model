import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PreNet(nn.Module):
    """A PreNet layer with low-rank factorization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        prefix_len: Optional[int] = 10,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.prefix_len = prefix_len
        self.linear_1 = nn.Linear(input_dim, rank)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(rank, output_dim * prefix_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x.view(-1, self.prefix_len, self.output_dim)


class Inverter:
    """Full embedding inverter pipeline."""

    def __init__(
        self,
        prenet: PreNet,
        decoder: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
    ):
        self.prenet = prenet
        self.decoder = decoder
        self.tokenizer = tokenizer

    def invert(
        self,
        x: torch.Tensor,
        max_len: Optional[int] = 30,
        temperature: Optional[float] = 0.1,
        repetition_penalty: Optional[float] = None,
    ):
        """Invert an embedding vector into text."""

        self.decoder.eval()
        self.prenet.eval()
        eos_id = self.tokenizer.eos_token_id

        temperature = max(temperature, 1e-9)

        with torch.no_grad():
            prefix = self.prenet(x.unsqueeze(0))
            generated = prefix
            generated_ids = []

            for _ in range(max_len):
                outs = self.decoder(inputs_embeds=generated)
                next_logits = outs.logits[:, -1, :].clone()

                if repetition_penalty is not None and repetition_penalty != 1.0:
                    repetition_penalty = max(1.0, repetition_penalty)
                    scores = next_logits.squeeze(0)
                    for prev_id in set(generated_ids):
                        if scores[prev_id] < 0:
                            scores[prev_id] = scores[prev_id] * repetition_penalty
                        else:
                            scores[prev_id] = scores[prev_id] / repetition_penalty
                    next_logits = scores.unsqueeze(0)

                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                token_id = next_id.item()
                generated_ids.append(next_id)

                if token_id == eos_id:
                    break

                next_embed = self.decoder.transformer.wte(next_id)
                generated = torch.cat([generated, next_embed], dim=1)

        gen_ids = torch.cat(generated_ids, dim=1)
        return self.tokenizer.decode(gen_ids[0].cpu().numpy(), skip_special_tokens=True)


def get_encoder(model_id: str, trainable: Optional[bool] = False):
    """Gets SentenceTransformer sentence-level embedding model."""
    model = SentenceTransformer(model_id)
    model.to(device)
    if not trainable:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model


def get_gpt2_decoder(trainable: Optional[bool] = False):
    """Gets GPT-2 (124M) decoder-only transformer model and tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    if not trainable:
        model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, tokenizer

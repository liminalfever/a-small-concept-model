import torch
import torch.nn.functional as F
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from small_concept_model.model import SmallConceptModel
from small_concept_model.inverter import Inverter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pipeline:
    """Full SCM pipeline with encoder and decoder."""

    def __init__(
        self,
        encoder: SentenceTransformer,
        model: SmallConceptModel,
        inverter: Inverter,
    ):
        self.encoder = encoder
        self.model = model
        self.inverter = inverter

    @staticmethod
    def clean_out(text: str) -> str:
        """Cleans the model's output."""

        text = text.strip()
        return text[0].upper() + text[1:] if text else text

    def generate_stream(
        self,
        texts: List[str],
        max_future_steps: Optional[int] = 5,
        max_len_sentence: Optional[int] = 30,
        temperature: Optional[float] = 0.1,
        repetition_penalty: Optional[float] = 1.1,
        similarity_threshold: Optional[float] = 0.9,
    ):
        """Generates a streaming sequence of texts from past ones."""

        self.model.eval()

        encoded_input = self.encoder.encode(texts, convert_to_tensor=True).to(device)

        for vec in encoded_input:
            text = self.inverter.invert(
                vec, max_len_sentence, temperature, repetition_penalty
            )
            yield self.clean_out(text) + " "

        for _ in range(max_future_steps):
            res = self.model(encoded_input.unsqueeze(0))[:, -1, :]

            cos_sim = F.cosine_similarity(res, encoded_input[-1], dim=-1)
            if cos_sim.mean().item() > similarity_threshold:
                break

            encoded_input = torch.cat([encoded_input, res], dim=0)
            text = self.inverter.invert(
                res, max_len_sentence, temperature, repetition_penalty
            )
            yield self.clean_out(text) + " "

    def generate(
        self,
        texts: List[str],
        max_future_steps: Optional[int] = 5,
        max_len_sentence: Optional[int] = 30,
        temperature: Optional[float] = 0.1,
        repetition_penalty: Optional[float] = 1.1,
        similarity_threshold: Optional[float] = 0.9,
    ):
        """Generates a sequence of texts from past ones."""

        self.model.eval()

        encoded_input = self.encoder.encode(texts, convert_to_tensor=True).to(device)

        out_texts = []
        for vec in encoded_input:
            text = self.inverter.invert(
                vec, max_len_sentence, temperature, repetition_penalty
            )
            out_texts.append(self.clean_out(text))

        for _ in range(max_future_steps):
            res = self.model(encoded_input.unsqueeze(0))[:, -1, :]

            cos_sim = F.cosine_similarity(res, encoded_input[-1], dim=-1)
            if cos_sim.mean().item() > similarity_threshold:
                break

            encoded_input = torch.cat([encoded_input, res], dim=0)
            text = self.inverter.invert(
                res, max_len_sentence, temperature, repetition_penalty
            )
            out_texts.append(self.clean_out(text))

        return out_texts

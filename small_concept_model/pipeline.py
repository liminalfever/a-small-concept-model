import torch
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from small_concept_model.model import SmallConceptModel
from small_concept_model.inverter import Inverter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Pipeline:
    """Full SCM pipeline with enoder and decoder."""

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
    def _generative_inference(
        model: SmallConceptModel,
        initial_sequence: torch.Tensor,
        n_future_steps: Optional[int] = 5,
        sigma_noise: Optional[float] = 0.0,
    ) -> torch.Tensor:
        """Generate a sequence of emeddings from past ones."""

        model.eval()

        prefix = initial_sequence.clone().unsqueeze(0).to(device)
        generated = prefix.clone()

        with torch.no_grad():
            for _ in range(n_future_steps):
                out = model(generated)
                last_embed = out[:, -1, :].unsqueeze(1)
                noise = torch.rand_like(last_embed) * sigma_noise
                last_embed = last_embed + noise
                generated = torch.cat([generated, last_embed], dim=1)

        return generated

    def generate(
        self,
        input_texts: List[str],
        n_future_steps: Optional[int] = 5,
        sigma_noise: Optional[float] = 0.0,
        temperature: Optional[float] = 0.1,
        max_len: Optional[int] = 30,
    ) -> List[str]:
        """Generates a sequence of texts from past ones."""

        if type(input_texts) != list:
            input_texts = [input_texts]

        encoded_inputs = self.encoder.encode(input_texts, convert_to_tensor=True).to(
            device
        )
        generated_seq = self._generative_inference(
            self.model, encoded_inputs, n_future_steps, sigma_noise=sigma_noise
        )

        new_sentences = []
        for v in generated_seq.squeeze(0):
            new_sentence = self.inverter.invert(v, max_len, temperature)
            new_sentences.append(new_sentence)

        return new_sentences

    def generate_stream(
        self,
        input_texts: List[str],
        n_future_steps: Optional[int] = 5,
        sigma_noise: Optional[float] = 0.0,
        temperature: Optional[float] = 0.1,
        max_len: Optional[int] = 30,
    ):
        """Generates a streaming sequence of texts from past ones."""

        if type(input_texts) != list:
            input_texts = [input_texts]

        encoded_inputs = self.encoder.encode(input_texts, convert_to_tensor=True).to(
            device
        )
        generated_seq = self._generative_inference(
            self.model, encoded_inputs, n_future_steps, sigma_noise=sigma_noise
        )

        for v in generated_seq.squeeze(0):
            yield self.inverter.invert(v, max_len, temperature)

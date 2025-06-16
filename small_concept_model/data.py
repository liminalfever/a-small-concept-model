import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer
from datasets import load_dataset
from typing import List, Tuple, Optional
from small_concept_model.utils import clean_text


class InverterDataset(Dataset):
    """Dataset class to train PreNet for embedding inversion."""

    def __init__(
        self, embeddings: torch.Tensor, input_ids_list: List[torch.Tensor], eos_id: int
    ):
        self.embeddings = embeddings
        self.eos_id = eos_id
        self.input_ids_list = input_ids_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        input_ids = self.input_ids_list[idx]
        eos_tensor = torch.tensor(
            [self.eos_id], dtype=torch.long, device=input_ids.device
        )
        input_ids_with_eos = torch.cat([input_ids, eos_tensor], dim=0)
        return embedding, input_ids_with_eos


class SCMDataset(Dataset):
    """Dataset class to train autoregressive SCM."""

    def __init__(self, data: torch.Tensor):
        assert data.ndim == 3
        self.seq_len = data.shape[1]
        self.data = data.float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]
        input_seq = seq[: self.seq_len - 1, :]
        target_seq = seq[1 : self.seq_len, :]
        return input_seq, target_seq


def get_bookcorpus_inverter(
    encoder: SentenceTransformer,
    tokenizer: GPT2Tokenizer,
    max_target_len: Optional[int] = 128,
    embed_batch_size: Optional[int] = 32,
    sample: Optional[float] = None,
    clean: Optional[bool] = False,
):
    """Gets Bookcorpus dataset (1M) for embedding inversion training."""

    data = load_dataset("francescoortame/bookcorpus-rand-1M", split="train")
    texts = data["text"]

    if sample:
        texts = texts[: int(sample * len(texts))]

    if clean:
        print("Cleaning texts...")
        texts = [clean_text(t) for t in texts]

    print("Tokenizing the texts...")
    input_ids = [
        tokenizer(
            s + tokenizer.eos_token,
            truncation=True,
            max_length=max_target_len,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze(0)
        for s in texts
    ]

    embeddings = encoder.encode(
        texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    return embeddings, input_ids


def get_bookcorpus_scm(
    encoder: SentenceTransformer,
    embed_batch_size: Optional[int] = 32,
    clean: Optional[bool] = False,
):
    """Gets Bookcorpus dataset (100k x 16) for autoregressive training."""

    data = load_dataset("francescoortame/bookcorpus-sorted-1M16x", split="train")
    flat_texts = [t for sublist in data["slice"] for t in sublist]

    if clean:
        flat_texts = [clean_text(t) for t in flat_texts]

    n_seqs = len(data)
    seq_len = len(data["slice"][0])
    d_embed = encoder.get_sentence_embedding_dimension()

    embeddings = encoder.encode(
        flat_texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    return embeddings.contiguous().view(n_seqs, seq_len, d_embed)

import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


class InversionTrainingDataset(Dataset):
    """Dataset class to train embedding inversion models."""

    def __init__(self, embeddings, input_ids_list):
        self.embeddings = embeddings
        self.input_ids_list = input_ids_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        input_ids = self.input_ids_list[idx]
        return embedding, input_ids


class SCMTrainingDataset(Dataset):
    """Dataset class to train autoregressive SCM."""

    def __init__(self, data: torch.Tensor):
        assert data.ndim == 3
        self.seq_len = data.shape[1]

        self.data = data.float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        seq = self.data[idx]
        input_seq = seq[: self.seq_len - 1, :]  # (seq_len - 1, embed_dim)
        target_seq = seq[1 : self.seq_len, :]  # (seq_len - 1, embed_dim)
        return input_seq, target_seq


def get_bookcorpus_for_inversion(
    encoder,
    tokenizer,
    max_target_length,
    train_batch_size=32,
    embed_batch_size=32,
    sample=None,
):
    """Get bookcorpus dataset (1M) of random sentences to train inversion models."""

    hf_data = load_dataset(
        "francescoortame/bookcorpus-rand-1M", split="train", trust_remote_code=True
    )
    hf_data = hf_data.train_test_split(0.1, seed=42)

    train_data, validation_data = hf_data["train"]["text"], hf_data["test"]["text"]

    if sample:
        train_data = train_data[: int(sample * len(train_data))]
        validation_data = validation_data[: int(sample * len(validation_data))]

    train_input_ids = [
        tokenizer(
            s,
            truncation=True,
            max_length=max_target_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze(0)
        for s in train_data
    ]
    validation_input_ids = [
        tokenizer(
            s,
            truncation=True,
            max_length=max_target_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze(0)
        for s in validation_data
    ]
    train_embeddings = encoder.encode(
        train_data,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    validation_embeddings = encoder.encode(
        validation_data,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )

    train_dataset = InversionTrainingDataset(train_embeddings, train_input_ids)
    validation_dataset = InversionTrainingDataset(
        validation_embeddings, validation_input_ids
    )

    return DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    ), DataLoader(validation_dataset, batch_size=train_batch_size, shuffle=False)


def get_bookcorpus_for_scm(
    encoder,
    train_batch_size=32,
    embed_batch_size=32,
):
    """Get bookcorpus dataset (100k x 16) of sorted sentences to train autoregressive models."""

    data = load_dataset("francescoortame/bookcorpus-sorted-100k16x", split="train")
    flat_texts = [t for sublist in data["slice"] for t in sublist]
    embeddings = encoder.encode(
        flat_texts,
        batch_size=embed_batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    reshaped_embeddings = embeddings.contiguous().view(100000, 16, 384)

    dataset = SCMTrainingDataset(reshaped_embeddings)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

    return dataloader
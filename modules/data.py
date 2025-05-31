from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class SentencePrefixDataset(Dataset):
    def __init__(self, texts, embedder, tokenizer, max_length):
        self.texts = texts
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sentence = self.texts[idx]
        
        embedding = self.embedder.encode(sentence, convert_to_tensor=True)
        encoded = self.tokenizer(sentence,
                                  truncation=True,
                                  max_length=self.max_length,
                                  padding='max_length',
                                  return_tensors='pt')
        input_ids = encoded.input_ids.squeeze(0)
        return embedding, input_ids


def get_dialog_dataloader(embedder, tokenizer, max_target_length, batch_size=32):
    hf_data = load_dataset('daily_dialog', split='train', trust_remote_code=True)
    sentences = [utt for dialog in hf_data['dialog'] for utt in dialog]
    sentences = sentences[:20000]

    dataset = SentencePrefixDataset(sentences, embedder, tokenizer, max_target_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_bookcorpus_dataloader(embedder, tokenizer, max_target_length, batch_size=32):
    hf_data = load_dataset('francescoortame/bookcorpus-rand-1M', split='train', trust_remote_code=True)
    hf_data = hf_data.train_test_split(0.1, seed=42)

    train_data, validation_data = hf_data["train"]["text"], hf_data["test"]["text"]

    train_dataset = SentencePrefixDataset(train_data, embedder, tokenizer, max_target_length)
    validation_dataset = SentencePrefixDataset(validation_data, embedder, tokenizer, max_target_length)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

class CachedSentenceDataset(Dataset):
    def __init__(self, embeddings, input_ids_list):
        self.embeddings = embeddings
        self.input_ids_list = input_ids_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        input_ids = self.input_ids_list[idx]
        return embedding, input_ids


def get_bookcorpus_dataloader(encoder, tokenizer, max_target_length, batch_size=32, embed_batch_size=32, sample=None):
    hf_data = load_dataset('francescoortame/bookcorpus-rand-1M', split='train', trust_remote_code=True)
    hf_data = hf_data.train_test_split(0.1, seed=42)

    train_data, validation_data = hf_data["train"]["text"], hf_data["test"]["text"]

    if sample:
        train_data = train_data[:int(sample * len(train_data))]
        validation_data = validation_data[:int(sample * len(validation_data))]

    train_input_ids = [
        tokenizer(s, truncation=True, max_length=max_target_length, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
        for s in train_data
    ]
    validation_input_ids = [
        tokenizer(s, truncation=True, max_length=max_target_length, padding="max_length", return_tensors="pt").input_ids.squeeze(0)
        for s in validation_data
    ]
    train_embeddings = encoder.encode(
        train_data, batch_size=embed_batch_size, show_progress_bar=True, convert_to_tensor=True
    )
    validation_embeddings = encoder.encode(
        validation_data, batch_size=embed_batch_size, show_progress_bar=True, convert_to_tensor=True
    )


    train_dataset = CachedSentenceDataset(train_embeddings, train_input_ids)
    validation_dataset = CachedSentenceDataset(validation_embeddings, validation_input_ids)

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

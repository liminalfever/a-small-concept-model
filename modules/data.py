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

def get_dataloader(embedder, tokenizer, max_target_length, batch_size=32):
    hf_data = load_dataset('daily_dialog', split='train', trust_remote_code=True)
    sentences = [utt for dialog in hf_data['dialog'] for utt in dialog]
    sentences = sentences[:20000]

    dataset = SentencePrefixDataset(sentences, embedder, tokenizer, max_target_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
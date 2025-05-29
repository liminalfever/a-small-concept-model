import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_encoder(model_id: str):
    model = SentenceTransformer(model_id)
    model.to(device)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

def get_gpt2_decoder():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    for param in model.parameters(): param.requires_grad = False
    return model, tokenizer
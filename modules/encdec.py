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

def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

def generative_inference(model, initial_sequence, n_future_steps, sigma_noise: float = None):
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

def vec_to_text(embedding, decoder, tokenizer, prenet, gen_len=50):
    """
    Given input text, encode it, generate prefix via PreNet, and autoregressively decode output text.
    """
    decoder.eval()
    prenet.eval()
    with torch.no_grad():
        prefix = prenet(embedding.unsqueeze(0))  # (1, prefix_len, model_dim)

        generated = prefix  # initial embeddings
        generated_ids = []
        for _ in range(gen_len):
            outputs = decoder(inputs_embeds=generated)
            next_logits = outputs.logits[:, -1, :]
            next_id = torch.argmax(next_logits, dim=-1).unsqueeze(-1)  # greedy
            generated_ids.append(next_id)
            next_embed = decoder.transformer.wte(next_id)
            generated = torch.cat([generated, next_embed], dim=1)

    gen_ids = torch.cat(generated_ids, dim=1)
    return tokenizer.decode(gen_ids[0].cpu().numpy(), skip_special_tokens=True)
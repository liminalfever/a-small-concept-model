import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
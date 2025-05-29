import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(prenet, decoder, train_dataloader, optimizer, loss_fct, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0
        for embeddings, input_ids in tqdm(train_dataloader, total=len(train_dataloader)):
            embeddings = embeddings.to(device)
            input_ids = input_ids.to(device)

            prefix_embeds = prenet(embeddings)

            token_embeds = decoder.transformer.wte(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds[:, :-1, :]], dim=1)

            outputs = decoder(inputs_embeds=inputs_embeds)
            logits = outputs.logits

            logits_tokens = logits[:, prenet.prefix_len:, :]
            labels = input_ids[:, 1:]

            B, Lm1, V = logits_tokens.size()
            loss = loss_fct(logits_tokens.reshape(-1, V), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
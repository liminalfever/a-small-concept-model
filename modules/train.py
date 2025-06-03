import torch
import torch.nn as nn
from modules.inverter import PreNet
from modules.scm import SmallConceptModel
from modules.utils import generate_causal_mask
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_inversion_model(
    prenet: PreNet,
    decoder: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    train_loader: DataLoader,
    validation_loader: Optional[DataLoader],
    lr: float = 1e-3,
    num_epochs: int = 5,
):
    """Train the PreNet model for embedding inversion."""
    
    optimizer = torch.optim.Adam(prenet.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    prenet.to(device)
    decoder.to(device)

    for epoch in range(num_epochs):
        prenet.train()
        decoder.train()
        total_train_loss = 0.0

        for embeddings, input_ids in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            embeddings = embeddings.to(device)
            input_ids  = input_ids.to(device)

            # forward
            prefix_embeds = prenet(embeddings)
            token_embeds  = decoder.transformer.wte(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds[:, :-1, :]], dim=1)

            outputs = decoder(inputs_embeds=inputs_embeds)
            logits  = outputs.logits[:, prenet.prefix_len:, :]  # shape [B, L, V]
            labels  = input_ids[:, 1:]                          # shifted targets

            B, L, V = logits.size()
            loss = loss_fct(logits.reshape(-1, V), labels.reshape(-1))

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        if validation_loader:
            prenet.eval()
            decoder.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for embeddings, input_ids in tqdm(validation_loader, desc=f"Epoch {epoch+1} [Valid]"):
                    embeddings = embeddings.to(device)
                    input_ids  = input_ids.to(device)

                    prefix_embeds = prenet(embeddings)
                    token_embeds  = decoder.transformer.wte(input_ids)
                    inputs_embeds = torch.cat([prefix_embeds, token_embeds[:, :-1, :]], dim=1)

                    outputs = decoder(inputs_embeds=inputs_embeds)
                    logits  = outputs.logits[:, prenet.prefix_len:, :]
                    labels  = input_ids[:, 1:]

                    B, L, V = logits.size()
                    loss = loss_fct(logits.reshape(-1, V), labels.reshape(-1))
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(validation_loader)
        
        else:
            avg_val_loss = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {avg_train_loss:.4f}   Val Loss: {avg_val_loss:.4f}")

def train_scm(
    model: SmallConceptModel,
    train_loader: DataLoader,
    max_seq_len: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    num_epochs: int = 5,
):
    """Train the SCM for next-embedding prediction."""

    causal_mask = generate_causal_mask(max_seq_len - 1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fct = nn.MSELoss(reduction="none")

    model.train()
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            # input_seq: (batch, 15, 384), target_seq: (batch, 15, 384)
            input_seq = input_seq.to(device)    # (BATCH_SIZE, SEQ_LEN−1,  EMBED_DIM)
            target_seq = target_seq.to(device)  # (BATCH_SIZE, SEQ_LEN−1,  EMBED_DIM)

            optimizer.zero_grad()
            # 6.6) Forward pass
            output = model(input_seq, causal_mask)
            # output shape: (batch, SEQ_LEN−1, EMBED_DIM)

            # 6.7) Compute loss: MSE over all positions
            # We do not want to predict beyond the provided target. Both output and target have shape (B, 15, 384).
            # So we can do a straightforward MSE.
            loss_tensor = loss_fct(output, target_seq)  # (B, 15, 384)
            loss = loss_tensor.mean()                    # scalar
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}]  "
                    f"Batch [{batch_idx+1}/{len(train_loader)}]  "
                    f"Loss: {loss.item():.6f}"
                )

        avg_epoch_loss = epoch_loss / n_batches
        print(f"*** Epoch {epoch} Complete.  Avg Loss = {avg_epoch_loss:.6f} ***")
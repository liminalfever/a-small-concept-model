import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from small_concept_model.data import InverterDataset, SCMDataset
from small_concept_model.inverter import PreNet
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from small_concept_model.model import SmallConceptModel
from tqdm import tqdm

def train_inverter(
    prenet: PreNet,
    decoder: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    train_dataset: InverterDataset,
    lr: float = 1e-3,
    weight_decay: Optional[float] = 1e-2,
    batch_size: Optional[int] = 32,
    num_epochs: int = 5,
):
    """Train the PreNet for embedding inversion."""

    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder.to(device)
    prenet.to(device)
    prenet.train()

    optimizer = torch.optim.Adam(prenet.parameters(), lr=lr)
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    for epoch in range(num_epochs):
        prenet.train()
        decoder.eval()
        total_train_loss = 0.0

        for embeddings, input_ids in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            embeddings = embeddings.to(device)
            input_ids  = input_ids.to(device)

            prefix_embeds = prenet(embeddings)
            token_embeds  = decoder.transformer.wte(input_ids)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds[:, :-1, :]], dim=1)

            outputs = decoder(inputs_embeds=inputs_embeds)
            logits  = outputs.logits[:, prenet.prefix_len:, :]  # shape [B, L, V]
            labels  = input_ids[:, 1:]                          # shifted targets

            B, L, V = logits.size()
            loss = loss_fct(logits.reshape(-1, V), labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_epoch_loss = total_train_loss / len(train_loader)

        print(f"*** Epoch {epoch} Complete.  Avg Loss = {avg_epoch_loss:.6f} ***")


def train_scm(
    model: SmallConceptModel,
    train_dataset: SCMDataset,
    lr: Optional[float] = 1e-3,
    weight_decay: Optional[float] = 1e-2,
    batch_size: Optional[int] = 32,
    num_epochs: Optional[int] = 1,
):
    """Train the SCM for next-embedding prediction."""

    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = torch.nn.MSELoss(reduction="none")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            optimizer.zero_grad()

            output = model(input_seq)
            loss_tensor = mse_loss(output, target_seq)
            loss = loss_tensor.mean()
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

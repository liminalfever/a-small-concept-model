import torch
import torch.nn as nn
from modules.data import get_dataloader
from modules.prenet import PreNet
from modules.encdec import get_encoder, get_gpt2_decoder
from modules.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resume_from: str = None

encoder = get_encoder("all-MiniLM-L6-v2")
decoder, tokenizer = get_gpt2_decoder()

train_dataloader = get_dataloader(encoder, tokenizer, 50, 64)

prenet = PreNet(
    input_dim=384,
    output_dim=768,
    bottleneck_dim=128,
    prefix_len=20
).to(device)

if resume_from:
    prenet.load_state_dict(torch.load(resume_from, map_location=device))

optimizer = torch.optim.Adam(prenet.parameters(), lr=1e-4)
loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

train(prenet, decoder, train_dataloader, optimizer, loss_fct, 5)

torch.save(prenet.state_dict(), 'saved_models/prenet_prefix_tuning.pth')
print("Saved PreNet to saved_models/prenet_prefix_tuning.pth")
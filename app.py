import streamlit as st
import torch
import nltk
from nltk.tokenize import sent_tokenize
from modules.scm import SmallConceptModel
from modules.prenet import PreNet
from modules.encdec import get_encoder, get_gpt2_decoder, generative_inference, vec_to_text

nltk.download('punkt_tab')

st.image("resources/repo-logo.png")

DATA_PATH     = "saved_models/train_seq_embeddings.pt"       # path to your (100k, 16, 384) NumPy file
LOAD_FROM     = "saved_models/scm_v01.pth"
BATCH_SIZE    = 32
NUM_EPOCHS    = 5
LEARNING_RATE = 1e-4
SEQ_LEN       = 16               # total length of each sequence 
EMBED_DIM     = 384              # dimension of each sentence embedding
D_MODEL       = 512              # model dimension (we keep it = EMBED_DIM)
NUM_LAYERS    = 3                # number of Transformer layers
NUM_HEADS     = 4                # number of attention heads
FFN_DIM       = 4 * D_MODEL      # feed‐forward “intermediate” dimension
DROPOUT       = 0.1
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_scm():
    model = SmallConceptModel(
        d_model=D_MODEL,
        embed_dim=EMBED_DIM,
        nhead=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dim_feedforward=FFN_DIM,
        dropout=DROPOUT,
        max_seq_len=SEQ_LEN
    ).to(DEVICE)

    model.load_state_dict(torch.load(LOAD_FROM, map_location=DEVICE))

    return model

def load_prenet():
    model = PreNet(
        input_dim=384,
        output_dim=768,
        bottleneck_dim=128,
        prefix_len=20
    ).to(DEVICE)
    
    model.load_state_dict(torch.load("saved_models/prenet_prefix_tuning_bookcorpus.pth", map_location=DEVICE))

    return model

if "scm" not in st.session_state:
    st.session_state["scm"] = load_scm()
if "encoder" not in st.session_state:
    st.session_state["encoder"] = get_encoder("all-MiniLM-L6-v2")
if "decoder" not in st.session_state:
    st.session_state["decoder"], st.session_state["tokenizer"] = get_gpt2_decoder()
if "prenet" not in st.session_state:
    st.session_state["prenet"] = load_prenet()

def stream_out(generated_seq):
    for vec in generated_seq.squeeze(0):
        text = vec_to_text(vec, st.session_state.decoder, st.session_state.tokenizer, st.session_state.prenet, 30)
        yield text


user_input = st.chat_input()
if user_input:
    input_sentences = sent_tokenize(user_input)
    embeddings = st.session_state.encoder.encode(input_sentences, convert_to_tensor=True).to(DEVICE)
    generated_seq = generative_inference(
        st.session_state["scm"], embeddings, 5, sigma_noise=0.01
    )

    st.write_stream(stream_out(generated_seq))


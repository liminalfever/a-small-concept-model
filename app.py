import streamlit as st
import torch
import nltk
from modules.scm import build_scm
from nltk.tokenize import sent_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "scm" not in st.session_state:
    with st.spinner("Building SCM Model..."):
        st.session_state.scm = build_scm()
        nltk.download("punkt_tab")

st.image("resources/repo-logo.png")

user_input = st.chat_input()
if user_input:
    input_sentences = sent_tokenize(user_input)

    st.write_stream(
        st.session_state.scm.generate_stream(
            input_sentences, 5, 0.01, 30
        )
    )
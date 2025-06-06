import streamlit as st
import torch
import nltk
from small_concept_model.auto import build_pipeline
from nltk.tokenize import sent_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "pipe" not in st.session_state:
    with st.spinner("Building SCM Model..."):
        st.session_state.pipe = build_pipeline(
            "scm_test_multilingual", "paraphrase_multilingual"
        )
        nltk.download("punkt_tab")

st.image("resources/repo-logo.png")

user_input = st.chat_input()
if user_input:
    input_sentences = sent_tokenize(user_input)

    st.write_stream(
        st.session_state.pipe.generate_stream(
            input_sentences,
            n_future_steps=5,
            sigma_noise=0.0,
            temperature=0.0,
            repetition_penalty=1.1,
            max_len=30
        )
    )

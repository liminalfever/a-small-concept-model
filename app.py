import streamlit as st
import torch
import nltk
from small_concept_model.auto import build_pipeline
from nltk.tokenize import sent_tokenize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if "pipe" not in st.session_state:
    with st.spinner("Building SCM Model..."):
        st.session_state.pipe = build_pipeline(
            "scm_multilingual", "inverter_multilingual"
        )
        nltk.download("punkt_tab")

st.image("resources/repo-logo.png")

with st.sidebar:
    st.slider("Temperature", 0.0, 1.0, 0.1, step=0.1, key="temperature")
    st.slider("Max Future Steps", 1, 10, 5, step=1, key="max_future_steps")
    st.number_input("Max Length Sentences", 1, 50, 30, step=1, key="len_sentences")
    st.slider("Similarity Threshold", 0.0, 1.0, 0.9, step=0.05, key="similarity_threshold")

user_input = st.chat_input()
if user_input:
    input_sentences = sent_tokenize(user_input)

    st.write_stream(
        st.session_state.pipe.generate_stream(
            input_sentences,
            max_future_steps=st.session_state["max_future_steps"],
            max_len_sentence=st.session_state["len_sentences"],
            temperature=st.session_state["temperature"],
            repetition_penalty=1.1,
            similarity_threshold=st.session_state["similarity_threshold"],
        )
    )

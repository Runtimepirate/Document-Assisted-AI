import os
import io
import re
from typing import List, Tuple

import streamlit as st
from transformers import pipeline
import PyPDF2

# Set environment variable to avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------- Sidebar -------------------- #
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/Runtimepirate/About_me/main/Profile_pic.jpg",
        width=200,
    )
    st.markdown(
        "## **Mr. Aditya Katariya [[Resume](https://drive.google.com/file/d/1Vq9-H1dl5Kky2ugXPIbnPvJ72EEkTROY/view?usp=drive_link)]**"
    )
    st.markdown(" *College - Noida Institute of Engineering and Technology, U.P*")
    st.markdown("----")
    st.markdown("## Contact Details:-")
    st.markdown("📫 *[Prasaritation@gmail.com](mailto:Prasaritation@gmail.com)*")
    st.markdown("💼 *[LinkedIn](https://www.linkedin.com/in/adityakatariya/)*")
    st.markdown("💻 *[GitHub](https://github.com/Runtimepirate)*")
    st.markdown("----")
    st.markdown("**AI & ML Enthusiast**")
    st.markdown(
        "Passionate about solving real-world problems using data science and customer analytics. Always learning and building smart, scalable AI solutions."
    )
    st.markdown("----")
    mode = st.radio("Choose Mode:", ["Ask Anything", "Challenge Me"], key="mode")

# -------------------- Title & Description -------------------- #
st.title("📚 Document‑Aware Assistant")

st.markdown(
    """
This assistant **reads your uploaded PDF or TXT document**, produces a *≤150‑word* summary, answers your questions with paragraph‑level justification, **generates logic‑based questions**, and evaluates your responses.
"""
)

# -------------------- Model Loading -------------------- #
@st.cache_resource(show_spinner=True)
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    qa = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
    return summarizer, qa

summarizer, qa_pipeline = load_models()

# -------------------- Helpers -------------------- #
def extract_text_from_pdf(uploaded_file: io.BytesIO) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return ""

def split_into_sentences(text: str) -> List[str]:
    # Simple regex-based sentence splitter that preserves paragraph structure
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.replace("\n", " ").strip() for s in sentences if s.strip()]

def chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    token_count = 0

    for sent in sentences:
        num_tokens = len(sent.split())
        if token_count + num_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            current = []
            token_count = 0
        current.append(sent)
        token_count += num_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks

# (rest of the file remains unchanged)

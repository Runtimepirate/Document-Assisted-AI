import os
import io
from typing import List, Tuple

import streamlit as st
from transformers import pipeline
import PyPDF2
import nltk
import nltk.data

# Force-download NLTK punkt to a safe Streamlit Cloud location
@st.cache_resource
def download_nltk():
    nltk_data_path = "/tmp/nltk_data"
    if not os.path.exists(nltk_data_path):
        os.makedirs(nltk_data_path)
    nltk.data.path.append(nltk_data_path)
    nltk.download("punkt", download_dir=nltk_data_path, quiet=True)

download_nltk()

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

def chunk_text(text: str, max_tokens: int = 450) -> List[str]:
    sentences = nltk.sent_tokenize(text)
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

def summarize_large_document(text: str, chunk_size: int = 800, max_words: int = 150) -> str:
    paragraphs = chunk_text(text, max_tokens=chunk_size)
    partial_summaries = []
    for para in paragraphs:
        try:
            summary = summarizer(
                para,
                max_length=80,
                min_length=20,
                do_sample=False
            )[0]['summary_text']
            partial_summaries.append(summary)
        except Exception:
            continue
    combined = " ".join(partial_summaries)
    try:
        final_summary = summarizer(
            combined[:4000],
            max_length=max_words,
            min_length=40,
            do_sample=False
        )[0]['summary_text']
        return final_summary
    except Exception:
        return combined[:max_words * 5]

def get_best_answer(question: str, chunks: List[str]) -> Tuple[str, int, int, float, str]:
    best = {"score": -float("inf")}
    for chunk in chunks:
        try:
            answer = qa_pipeline(question=question, context=chunk)
            if answer["score"] > best["score"] and answer["answer"].strip():
                best = {
                    "answer": answer["answer"],
                    "score": answer["score"],
                    "start": answer["start"],
                    "end": answer["end"],
                    "context": chunk,
                }
        except Exception:
            continue
    if best["score"] == -float("inf"):
        return "", 0, 0, 0.0, ""
    return (
        best["answer"],
        best["start"],
        best["end"],
        best["score"],
        best["context"],
    )

def highlight_answer(context: str, start: int, end: int) -> str:
    return context[:start] + " **" + context[start:end] + "** " + context[end:]

def generate_static_questions() -> List[str]:
    return [
        "What is the main topic of the document?",
        "Summarize the methodology described.",
        "What are the key findings or conclusions?",
    ]

# -------------------- Main Workflow -------------------- #

def main():
    uploaded = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"], key="uploader")

    if uploaded:
        doc_text = extract_text(uploaded)
        st.session_state["doc_text"] = doc_text

        st.subheader("🔎 Auto Summary (≤ 150 words)")
        try:
            summary = summarize_large_document(doc_text)
            st.write(summary)
        except Exception as e:
            st.error(f"Summarization failed: {e}")

        if "chunks" not in st.session_state:
            st.session_state["chunks"] = chunk_text(doc_text)

        if mode == "Ask Anything":
            st.subheader("💬 Ask Anything")
            question = st.text_input("Ask a question about the document:", key="user_question")
            if st.button("Submit Question", key="submit_question") and question:
                with st.spinner("Finding answer..."):
                    ans, start, end, score, context = get_best_answer(
                        question, st.session_state["chunks"]
                    )
                if ans:
                    st.markdown(f"**Answer:** {ans}")
                    justification = highlight_answer(context, start, end)
                    st.caption(f"Justification: …{justification[:300]}…")
                    st.caption(f"Confidence Score: {score:.3f}  |  Paragraph tokens: {len(context.split())}")
                else:
                    st.warning("Sorry, I couldn't find an answer in the document.")

        elif mode == "Challenge Me":
            st.subheader("🎯 Challenge Me")
            if "logic_questions" not in st.session_state:
                st.session_state["logic_questions"] = generate_static_questions()
                st.session_state["user_answers"] = ["" for _ in st.session_state["logic_questions"]]

            for idx, q in enumerate(st.session_state["logic_questions"]):
                st.text_input(f"Q{idx+1}: {q}", key=f"logic_q_{idx}")

            if st.button("Submit Answers", key="submit_logic"):
                st.markdown("----")
                for idx, q in enumerate(st.session_state["logic_questions"]):
                    user_ans = st.session_state.get(f"logic_q_{idx}", "").strip()
                    correct, start, end, score, context = get_best_answer(
                        q, st.session_state["chunks"]
                    )
                    st.markdown(f"**Q{idx+1} Evaluation:**")
                    st.write(f"*Your Answer*: {user_ans or '—'}")
                    st.write(f"*Expected Answer*: {correct or 'Not found in document'}")
                    if correct:
                        justification = highlight_answer(context, start, end)
                        st.caption(f"Justification: …{justification[:300]}…")
                        st.caption(f"Confidence Score: {score:.3f}")
                    st.markdown("----")

    else:
        st.info("Please upload a PDF or TXT document to begin.")

if __name__ == "__main__":
    main()

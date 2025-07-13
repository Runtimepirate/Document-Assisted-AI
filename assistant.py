# # ===============================
# # Document‑Aware GenAI Assistant
# # Streamlit application (single‑file)
# # Author: ChatGPT (adapted for Aditya)
# # ===============================
# # ⚠️  Before running, make sure you have an OpenAI API key set as the env variable
# #     OPENAI_API_KEY. Install the dependencies listed at the bottom of this file.
# # ===============================

# from __future__ import annotations

# import os
# import tempfile
# from pathlib import Path
# from typing import List, Tuple, Dict

# import streamlit as st
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import PyPDFLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# # ---------- CONFIG ----------
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 150
# MODEL_NAME = "gpt-4o-mini"  # adjust if needed
# TEMPERATURE = 0
# SUMMARY_MAX_WORDS = 150

# # ---------- INITIALISE LLM & EMBEDDINGS ----------
# @st.cache_resource(show_spinner=False)
# def get_llm():
#     return ChatOpenAI(model_name=MODEL_NAME, temperature=TEMPERATURE)

# @st.cache_resource(show_spinner=False)
# def get_embeddings():
#     return OpenAIEmbeddings(model="text-embedding-3-small")

# # ---------- DOCUMENT LOADING & INDEXING ----------

# def load_document(uploaded_file) -> List[str]:
#     """Load PDF or TXT into raw text pages."""
#     with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
#         tmp.write(uploaded_file.getbuffer())
#         tmp_path = tmp.name
#     if uploaded_file.type == "application/pdf":
#         loader = PyPDFLoader(tmp_path)
#         pages = loader.load()
#     else:
#         loader = TextLoader(tmp_path, encoding="utf-8")
#         pages = loader.load()
#     os.unlink(tmp_path)
#     return pages

# @st.cache_resource(show_spinner=True)
# def split_and_embed(pages) -> Tuple[FAISS, List]:
#     splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#     splits = splitter.split_documents(pages)
#     vectordb = FAISS.from_documents(splits, get_embeddings())
#     return vectordb, splits

# # ---------- SUMMARISATION ----------

# def summarise_document(llm, pages) -> str:
#     raw_text = "\n".join([page.page_content for page in pages])
#     prompt = (
#         f"Summarise the following document in no more than {SUMMARY_MAX_WORDS} words.\n\n" + raw_text
#     )
#     summary = llm.invoke(prompt).content
#     # Ensure <= 150 words (simple enforcement)
#     trimmed = " ".join(summary.strip().split()[:SUMMARY_MAX_WORDS])
#     return trimmed

# # ---------- QUESTION‑ANSWERING CHAIN ----------

# def get_qa_chain(llm, vectordb):
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
#         return_source_documents=True,
#     )
#     return qa_chain

# # ---------- LOGIC‑BASED QUESTION GENERATION ----------

# def generate_logic_questions(llm, context_docs, num_q: int = 3) -> List[str]:
#     context = "\n".join([doc.page_content for doc in context_docs[:10]])  # first 10 chunks for diversity
#     prompt = (
#         "You are an examiner. Generate "
#         f"{num_q} challenging, logic‑based comprehension questions based on the context below.\n\n"
#         f"Context:\n{context}\n\n"
#         "Number them 1‑3. Keep each question concise (max 25 words)."
#     )
#     questions = llm.invoke(prompt).content.strip().split("\n")
#     # Clean list
#     questions = [q.lstrip("1234567890. ").strip() for q in questions if q.strip()]
#     return questions[:num_q]

# # ---------- ANSWER GRADING ----------

# def grade_answer(llm, question: str, correct_answer: str, user_answer: str) -> Dict:
#     grading_prompt = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(
#             "You are a strict examiner. You are given a question, the model's reference answer, and the user's answer."
#             " Evaluate whether the user's answer is entirely correct, partially correct, or incorrect, using ONLY the reference answer as ground truth."
#             " Output strictly in JSON with keys correctness (Correct / Partial / Incorrect) and feedback (max 40 words)."
#         ),
#         HumanMessagePromptTemplate.from_template(
#             "Question: {question}\nReference Answer: {ref_ans}\nUser Answer: {user_ans}"
#         ),
#     ])
#     resp = llm.invoke(grading_prompt.format(question=question, ref_ans=correct_answer, user_ans=user_answer)).content
#     try:
#         import json
#         result = json.loads(resp)
#     except Exception:
#         # Fallback parsing
#         result = {"correctness": "Unknown", "feedback": resp}
#     return result

# # ---------- STREAMLIT UI ----------

# st.set_page_config(page_title="Document‑Aware Assistant", layout="wide")
# st.title("📄🔎 Document‑Aware GenAI Assistant")

# # Sidebar – File upload
# uploaded = st.sidebar.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if uploaded:
#     st.write("### Document Loaded")
#     # Build vector store & summary once per upload
#     vectordb, splits = split_and_embed(load_document(uploaded))
#     llm = get_llm()
#     if "summary" not in st.session_state or st.session_state.get("last_file") != uploaded.name:
#         summary = summarise_document(llm, splits)
#         st.session_state.summary = summary
#         st.session_state.vectordb = vectordb
#         st.session_state.splits = splits
#         st.session_state.last_file = uploaded.name
#         st.session_state.qa_chain = get_qa_chain(llm, vectordb)
#         st.session_state.logic_qs = generate_logic_questions(llm, splits)

#     st.info(st.session_state.summary)

#     mode = st.radio("Choose mode", ["Ask Anything", "Challenge Me"], horizontal=True)

#     # ----------- ASK ANYTHING MODE -----------
#     if mode == "Ask Anything":
#         query = st.text_input("Ask a question about the document: ")
#         if query:
#             with st.spinner("Thinking…"):
#                 response = st.session_state.qa_chain(query)
#             answer = response["result"]
#             sources = response["source_documents"]
#             st.markdown(f"**Answer:** {answer}")
#             if sources:
#                 with st.expander("Supporting snippets"):
#                     for i, src in enumerate(sources, 1):
#                         st.markdown(f"**Snippet {i}:** …{src.page_content[:350]}…")

#     # ----------- CHALLENGE ME MODE -----------
#     else:
#         st.write("### Your challenge questions")
#         for idx, q in enumerate(st.session_state.logic_qs, 1):
#             user_ans = st.text_input(f"{idx}. {q}", key=f"ans_{idx}")
#             if user_ans:
#                 # Compute reference answer on‑the‑fly
#                 ref_resp = st.session_state.qa_chain(q)
#                 ref_ans = ref_resp["result"]
#                 grade = grade_answer(get_llm(), q, ref_ans, user_ans)
#                 st.markdown(f"**Your answer is:** {grade['correctness']}")
#                 st.caption(grade["feedback"])
#                 with st.expander("Reference & justification"):
#                     st.markdown(f"**Reference answer:** {ref_ans}")
#                     for j, src in enumerate(ref_resp["source_documents"], 1):
#                         st.markdown(f"*Snippet {j}: …{src.page_content[:300]}…*")

# # ---------- FOOTER & REQUIREMENTS ----------

# st.sidebar.markdown("---")
# st.sidebar.markdown("ℹ️ **Instructions**\n\n- Obtain an OpenAI API key and set it as OPENAI_API_KEY.\n- Install dependencies with:`pip install -r requirements.txt`\n- Run with:`streamlit run doc_aware_assistant_app.py`\n")

# # Include requirements inline for convenience
# if st.sidebar.checkbox("Show requirements.txt"):
#     st.code(
#         """\nopenai>=1.12.0\nstreamlit>=1.35.0\nlangchain>=0.1.17\nfaiss‑cpu>=1.7.4\nPyPDF2>=3.0.1\npypdf<4\n""",
#         language="text",
#     )



import streamlit as st
from transformers import pipeline
import PyPDF2
import io

# Load models once
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return summarizer, qa

summarizer, qa = load_models()

def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    return ""

st.title("Document-Aware GenAI Assistant (Hugging Face Spaces)")

st.header("1. Upload PDF or TXT Document")
uploaded = st.file_uploader("Choose a file", type=["pdf", "txt"])
if uploaded:
    doc_text = extract_text(uploaded)
    st.session_state["doc_text"] = doc_text

    st.subheader("Auto Summary (≤ 150 words)")
    summary = summarizer(doc_text[:4000], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    st.markdown(f"*{summary}*")

    st.header("2. Choose Mode")
    mode = st.radio("Select interaction mode:", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        question = st.text_input("Ask a question about the document:")
        if st.button("Submit Question") and question:
            answer = qa(question=question, context=doc_text)
            st.markdown(f"**Answer:** {answer['answer']}")
            # Optionally show supporting snippet
            snippet = doc_text[max(0, answer['start']-50):answer['end']+50]
            st.caption(f"Context: ...{snippet}...")

    elif mode == "Challenge Me":
        st.info("Logic-based question generation requires a generative model. For demo, enter your own questions or use a simple template.")
        # Example: Generate sample questions (can be replaced with a QG model)
        sample_questions = [
            "What is the main topic of the document?",
            "Summarize the methodology described.",
            "What are the key findings or conclusions?"
        ]
        user_answers = []
        for i, q in enumerate(sample_questions):
            ans = st.text_input(f"Q{i+1}: {q}", key=f"challenge_{i}")
            user_answers.append((q, ans))
        if st.button("Submit Answers"):
            for i, (q, user_ans) in enumerate(user_answers):
                if user_ans:
                    answer = qa(question=q, context=doc_text)
                    correct = answer['answer']
                    st.markdown(f"**Q{i+1} Evaluation:**")
                    st.markdown(f"- **Your Answer:** {user_ans}")
                    st.markdown(f"- **Expected Answer:** {correct}")
                    snippet = doc_text[max(0, answer['start']-50):answer['end']+50]
                    st.caption(f"Justification: ...{snippet}...")


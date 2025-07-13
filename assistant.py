import streamlit as st
from transformers import pipeline
import PyPDF2

with st.sidebar:
    st.image("https://raw.githubusercontent.com/Runtimepirate/About_me/main/Profile_pic.jpg", width=200)
    st.markdown("## **Mr. Aditya Katariya [[Resume](https://drive.google.com/file/d/1Vq9-H1dl5Kky2ugXPIbnPvJ72EEkTROY/view?usp=drive_link)]**")
    st.markdown(" *College - Noida Institute of Engineering and Technology, U.P*")
    st.markdown("----")
    st.markdown("## Contact Details:-")
    st.markdown("📫 *[Prasaritation@gmail.com](mailto:Prasaritation@gmail.com)*")
    st.markdown("💼 *[LinkedIn](https://www.linkedin.com/in/adityakatariya/)*")
    st.markdown("💻 *[GitHub](https://github.com/Runtimepirate)*")
    st.markdown("----")
    st.markdown("**AI & ML Enthusiast**")
    st.markdown(
        """
        Passionate about solving real-world problems using data science and customer analytics. Always learning and building smart, scalable AI solutions.
        """
    )
    st.markdown("----")
    mode = st.radio("Choose Mode:", ["Ask Anything", "Challenge Me"])


st.title("Document-Aware GenAI Assistant")

st.markdown("""
This assistant reads your uploaded PDF or TXT document, summarizes it, answers your questions, and can quiz you with logic-based questions.
""")

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

uploaded = st.file_uploader("Upload PDF or TXT Document", type=["pdf", "txt"])

if uploaded:
    doc_text = extract_text(uploaded)
    st.session_state["doc_text"] = doc_text

    st.subheader("Auto Summary (≤ 150 words)")
    try:
        summary = summarizer(doc_text[:4000], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.markdown(f"*{summary}*")
    except Exception as e:
        st.error(f"Summarization failed: {e}")

    if mode == "Ask Anything":
        st.subheader("Ask Anything")
        question = st.text_input("Ask a question about the document:")
        if st.button("Submit Question") and question:
            try:
                answer = qa(question=question, context=doc_text)
                st.markdown(f"**Answer:** {answer['answer']}")
                snippet = doc_text[max(0, answer['start']-50):answer['end']+50]
                st.caption(f"Context: ...{snippet}...")
            except Exception as e:
                st.error(f"Question answering failed: {e}")

    elif mode == "Challenge Me":
        st.subheader("Challenge Me")
        st.info("Logic-based question generation is demoed below. For production, consider using a question generation model like T5.")

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
                    try:
                        answer = qa(question=q, context=doc_text)
                        correct = answer['answer']
                        st.markdown(f"**Q{i+1} Evaluation:**")
                        st.markdown(f"- **Your Answer:** {user_ans}")
                        st.markdown(f"- **Expected Answer:** {correct}")
                        snippet = doc_text[max(0, answer['start']-50):answer['end']+50]
                        st.caption(f"Justification: ...{snippet}...")
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

else:
    st.info("Please upload a PDF or TXT document to begin.")

st.markdown("---")

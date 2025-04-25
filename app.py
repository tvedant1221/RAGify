import streamlit as st
from utils import load_pdf, get_embeddings, ask_question

st.set_page_config(page_title = "PDF Q&A", layout = 'wide')
st.title("PDF Q&A")

pdf_file = st.file_uploader('Upload a PDF file', type = 'pdf')
if pdf_file:
    with st.spinner('Running PDF...'):
        text = load_pdf(pdf_file)
        vs = get_embeddings(text)
        st.success("PDF Uploaded...")

query = st.text_input("Ask your Question : ")

if query:
            with st.spinner('Thinking...'):
                answer = ask_question(vs, query)
                st.write('Answer : ', answer)

st.markdown("### 🔍 Retrieved context:")
for doc in docs:
    st.markdown(f"• {doc.page_content[:300]}...")

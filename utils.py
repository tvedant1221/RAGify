import fitz
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.docstore.document import Document

from huggingface_hub import login
import os
token = os.environ["HF_TOKEN"]
login(token=token)

model_id = 'mistralai/Mistral-7B-Instruct-v0.1'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map = 'auto', load_in_4bit = True)
qa_pipeline = pipeline('text-generation', model = model, tokenizer = tokenizer, max_new_tokens = 500)

# For small responses.
# qa_pipeline = pipeline('text2text-generation', model = 'google/flan-t5-small', max_new_tokens = 500)

embedder = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
llm = HuggingFacePipeline(pipeline = qa_pipeline)

def load_pdf(pdf_file):
    docs = fitz.open(stream = pdf_file.read(), filetype = 'pdf')
    text = '\n'.join([page.get_text() for page in docs])
    return text

def get_embeddings(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 100)
    chunks = splitter.split_text(text)
    docs = [Document(page_content = chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embedder)

def ask_question(vs, query):
    docs = vs.similarity_search(query, k = 3)
    chain = load_qa_chain(llm, chain_type = 'map_reduce')
    return chain.run(input_documents = docs, question = query)
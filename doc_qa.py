import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

st.title("Document-based Question Answering System (GenAI)")

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None

if uploaded_file and OPENAI_API_KEY:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.write("Document uploaded and text extracted.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Create or load Chroma vector store with document embeddings
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    st.write("Document processed with embeddings. Ready for querying.")

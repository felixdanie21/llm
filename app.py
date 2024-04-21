import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import os

load_dotenv('.env')

col1, col2 = st.columns([3, 4])

with col1:
    st.title("Layanan BPJS")
    
with col2:
    st.image("logo.png", width=70)

# Fungsi untuk membaca teks dari PDF dan menyimpannya ke dalam file teks
def save_text_from_pdf(pdf_path, text_file_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()

    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        text_file.write(text)

# Lokasi penyimpanan teks dari PDF
text_file_path_bpjs = "bpjs_text_bpjs.txt"
text_file_path_jkn = "bpjs_text_jkn.txt"

# Ganti dengan path file PDF yang sesuai
pdf_path_bpjs = "bpjs.pdf"
pdf_path_jkn = "jkn.pdf"

# Cek apakah file teks sudah ada, jika tidak, baca dan simpan teks dari PDF
if not os.path.exists(text_file_path_bpjs):
    save_text_from_pdf(pdf_path_bpjs, text_file_path_bpjs)

if not os.path.exists(text_file_path_jkn):
    save_text_from_pdf(pdf_path_jkn, text_file_path_jkn)

# Split the text into chunks

text_splitter_bpjs = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

text_splitter_jkn = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

with open(text_file_path_bpjs, 'r', encoding='utf-8') as text_file:
    text_bpjs = text_file.read()
with open(text_file_path_jkn, 'r', encoding='utf-8') as text_file:
    text_jkn = text_file.read()

chunks_bpjs = text_splitter_bpjs.split_text(text_bpjs)
chunks_jkn = text_splitter_jkn.split_text(text_jkn)

# Embeddings
embeddings = OpenAIEmbeddings()
knowledge_base_bpjs = FAISS.from_texts(chunks_bpjs, embeddings)
knowledge_base_jkn = FAISS.from_texts(chunks_jkn, embeddings)

# Pertanyaan
pertanyaan = st.text_input("Tentang BPJS atau JKN ya")

if pertanyaan:
    if "BPJS" in pertanyaan.upper():
        docs = knowledge_base_bpjs.similarity_search(pertanyaan)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=pertanyaan)

        st.write(response)
    elif "JKN" in pertanyaan.upper():
        docs = knowledge_base_jkn.similarity_search(pertanyaan)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=pertanyaan)

        st.write(response)
    else:
        st.write("Maaf, pertanyaan anda tidak terkait dengan BPJS atau JKN.")

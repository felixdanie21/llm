from PyPDF2 import PdfReader
import argparse
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

load_dotenv()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
args = vars(ap.parse_args())

pdf_reader = PdfReader(args['input'])
text =""
for page in pdf_reader.pages:
    text += page.extract_text()

#split menjadi chunk
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_text(text)

# embeddings
print("[INFO] Embedding Texts")
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks,embeddings)

#chat logic
print("[INFO] Starting Bot ..")
while True:
    pertanyaan = input("Anda > ")
    docs = knowledge_base.similarity_search(pertanyaan)

    llm = OpenAI()
    chain = load_qa_chain(llm,chain_type="stuff")
    with get_openai_callback() as cb :
        response = chain.run(input_documents=docs, question=pertanyaan)
    print(response)
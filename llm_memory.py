from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Set the data directory
DATA_PATH = r"C:\Users\Lenovo\OneDrive\Desktop\CSE299.9\medical recom sys\data"

def load_pdf_files(data):
    if not os.path.exists(data):
        print(f"Error: Directory {data} not found.")
        return []

    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    
    documents = loader.load()

    if not documents:
        print("No PDF files found.")
    
    return documents

# Load the documents
documents = load_pdf_files(DATA_PATH)


#print("Length of PDF pages:", len(documents))

# Creating Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
#print("Length of Text Chunks: ", len(text_chunks))

 # Loading Vector Embeddings Model 
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
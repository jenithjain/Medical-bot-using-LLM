from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Directory paths
DATA_PATH = "data"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Create vectorstore directory if it doesn't exist
if not os.path.exists("vectorstore"):
    os.makedirs("vectorstore")

def create_vector_db():
    # Load documents
    loader = DirectoryLoader(DATA_PATH,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                  chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and FAISS database
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
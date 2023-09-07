# LoadVectorize.py

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# access some online pdf, load, vectorize and commit to disk
def load_vectorize(embeddings):
    loader = OnlinePDFLoader("https://support.riverbed.com/bin/support/download?did=b42r9nj98obctctoq05bl2qlga&version=9.14.2a")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    # vectorise
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("./opdf_index")
    return db

# attempts to load vectorstore from disk
def load_db():
    embeddings = HuggingFaceEmbeddings()
    try:
        db = FAISS.load_local("./opdf_index", embeddings)
    except:
        print('no index on disk, creating new...')
        db = load_vectorize(embeddings)
    return db

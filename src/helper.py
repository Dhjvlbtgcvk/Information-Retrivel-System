from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
HF_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def get_pdf_text(pdf_docs):
    """Extract text from list of uploaded PDFs"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunk(text):
    """Split large text into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Convert chunks into embeddings and store in FAISS"""
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)
    return vector_store


def get_conversational_chain(vector_store):
    """Build the conversational retrieval chain"""
    llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_KEY, temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # fixed typo here
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

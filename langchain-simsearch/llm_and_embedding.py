import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

def llm():
    model = "gpt-4.1-nano"
    llm = ChatOpenAI(model_name=model, temperature=0, openai_api_key=openai_api_key)
    return llm

def openai_embedding():
    embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)
    # Alternatively, using ChromaDB's embedding function
    return embedding

def openai_chroma_embedding():
    embedding = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name="text-embedding-3-small")
    return embedding
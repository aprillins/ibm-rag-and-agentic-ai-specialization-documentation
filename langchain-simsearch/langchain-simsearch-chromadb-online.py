from chromadb import CloudClient
import os
from dotenv import load_dotenv
from llm_and_embedding import openai_chroma_embedding
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

chromadb_api_key = os.getenv("CHROMA_API_KEY")
chromadb_host = os.getenv("CHROMA_HOST")
chromadb_tenant = os.getenv("CHROMA_TENANT")
chromadb_database = os.getenv("CHROMA_DATABASE")

# Initialize ChromaDB Cloud Client
client = CloudClient(
    cloud_host=chromadb_host,
    api_key=chromadb_api_key,
    tenant=chromadb_tenant,
    database=chromadb_database
)

collection_name = "company_policy"
loader = TextLoader("companypolicies.txt")
documents = loader.load()
# texts = [doc.page_content for doc in documents]
text = documents[0].page_content
embeddings_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")

print(f"{type(text)} -{type(documents)}")

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
splitted_docs = splitter.split_text(text)

print(f"Number of splitted documents: {len(splitted_docs)}")

def add_collection(collection):
    ids = [f"policy_{index + 1}" for index, _ in enumerate(splitted_docs)]

    collection.add(
        ids=ids,
        documents=splitted_docs,
    )

    all_items = collection.get()
    print("Collection contents:")
    print(f"Number of documents: {len(all_items['documents'])}")
try:
    collection = client.get_collection(name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
    add_collection(collection)
except Exception as e:
    collection = client.create_collection(
        name=collection_name, 
        embedding_function=embeddings_function,
        metadata={"description": "company policy"}
    )

    add_collection(collection)


import chromadb 
import os
from pets_data import pets
from dotenv import load_dotenv
from chromadb import CloudClient
from chromadb.utils import embedding_functions

load_dotenv()
chromadb_api_key = os.getenv("CHROMA_API_KEY")
chromadb_host = os.getenv("CHROMA_HOST")
chromadb_tenant = os.getenv("CHROMA_TENANT")
chromadb_database = os.getenv("CHROMA_DATABASE")

#ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
# client = chromadb.Client()

client = CloudClient(
    cloud_host = chromadb_host,
    api_key = chromadb_api_key,
    tenant = chromadb_tenant,
    database = chromadb_database
)
print(client.heartbeat())
# collection_name = "pets_collectiosn"
collection_name = "pets_collection_openai_embedding"


def main():
    try:
        collection = client.get_or_create_collection(
            name=collection_name, 
            embedding_function=ef,
            metadata={"description": "A collection for pets"},
            configuration={
                # "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )
        all_items = collection.get()
        # Logging the retrieved items to the console for inspection or debugging
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        perform_advanced_search(collection, all_items)
        pass
    except Exception as error:
        print("An error occurred:", str(error))

def perform_advanced_search(collection, all_items):
    try:
        # Advanced search operations will be placed here
        breed = "doberman"
        print("=== Similarity Search Examples ===")
        # Example 1: Search for Beagles
        print(f"\n1. Searching 2-3 years old {breed}")
        query_text = f"{breed}, 2-3 years old"
        results = collection.query(
            query_texts=[query_text],
            n_results=5
        )
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
            results['ids'][0], results['documents'][0], results['distances'][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Species: {metadata['species']}, Breed: {metadata['breed']}")
            print(f"     Document: {document[:100]}...")

        print("\n=== Metadata Filtering Examples ===")
        # Example 1: Filter by Location
        location = "Atlanta"
        print(f"\n2. Finding all pets located in {location}:")
        results = collection.get(
            where={"location": location}
        )
        print(f"Found {len(results['ids'])} pets:")
        for i, doc_id in enumerate(results['ids']):
            metadata = results['metadatas'][i]
            print(f"  - {metadata['name']}: {metadata['species']} | {metadata['breed']} | ({metadata['weight_kg']} kg)")

        pass
    except Exception as error:
        print(f"Error in advanced search: {error}")


if __name__ == "__main__":
    main()
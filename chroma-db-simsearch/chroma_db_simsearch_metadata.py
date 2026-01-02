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

# ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2") # using sentence transformer is slow because it connects to Huggingface first
ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")

# client = chromadb.Client()

client = CloudClient(
    cloud_host = chromadb_host,
    api_key = chromadb_api_key,
    tenant = chromadb_tenant,
    database = chromadb_database
)
print(client.heartbeat())
# collection_name = "pets_collection"
collection_name = "pets_collection_openai_embedding"

def main():
    try:
        collection = client.get_or_create_collection(
            name=collection_name, 
            embedding_function=ef,
            metadata={"description": "A collection for pets"},
            configuration={
                "hnsw": {"space": "cosine"},
                "embedding_function": ef
            }
        )

        pets_data = pets
        pet_documents = []
        for pet in pets_data:
            document = f"{pet['species']} named {pet['name']}, age {pet['age']} years, breed {pet['breed']}. "
            document += f"Color: {pet['color']}, weight: {pet['weight_kg']} kg. Located in {pet['location']}. "
            document += f"Adoption status: {pet['adoption_status']}."
            pet_documents.append(document)

        collection.add(
            # Extracting pet IDs to be used as unique identifiers for each record
            ids=[pet["id"] for pet in pets],
            # Using the comprehensive text documents we created for pets
            documents=pet_documents,
            # Adding metadata for filtering and search tailored for pets
            metadatas=[{
                "name": pet["name"],
                "species": pet["species"],
                "breed": pet["breed"],
                "age": pet["age"],
                "color": pet["color"],
                "weight_kg": pet["weight_kg"],
                "location": pet["location"],
                "adoption_status": pet["adoption_status"]
            } for pet in pets]
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

        print("=== Similarity Search Examples ===")
        # Example 1: Search for Beagles
        print("\n1. Searching 2-3 years old Beagle")
        query_text = "Beagle, 2-3 years old"
        results = collection.query(
            query_texts=[query_text],
            n_results=3
        )
        print(f"Query: '{query_text}'")
        for i, (doc_id, document, distance) in enumerate(zip(
            results['ids'][0], results['documents'][0], results['distances'][0]
        )):
            metadata = results['metadatas'][0][i]
            print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
            print(f"     Species: {metadata['species']}, Breed: {metadata['breed']}")
            print(f"     Document: {document[:100]}...")

        pass
    except Exception as error:
        print(f"Error in advanced search: {error}")


if __name__ == "__main__":
    main()
# chromadb==0.3.23
# sentence-transformers==5.2.0
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection_name = "my_simsearch_collection"

def main():
    try:
        collection = client.create_collection(
            name=collection_name, 
            embedding_function=ef,
            metadata={"description": "A collection for groceries"}
        )
        print(f"Collection created with name:", collection.name)

        texts = [
            'fresh red apples',
            'organic bananas',
            'ripe mangoes',
            'whole wheat bread',
            'farm-fresh eggs',
            'natural yogurt',
            'frozen vegetables',
            'grass-fed beef',
            'free-range chicken',
            'fresh salmon fillet',
            'aromatic coffee beans',
            'pure honey',
            'golden apple',
            'red fruit'
        ]

        # Create a list of unique IDs for each text item in the 'texts' array
        # Each ID follows the format 'food_<index>', where <index> starts from 1
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]

        collection.add(
            documents=texts,
            ids=ids,
            metadatas=[{"source": "grocery_store", "category": "food"} for _ in texts]
        )

        all_items = collection.get()
        print("Collection contents:")
        print(f"Number of documents: {len(all_items['documents'])}")

        # Function to perform a similarity search in the collection
        def perform_similarity_search(collection, all_items):
            try:
                # Perform a query to search for the most similar documents to the 'query_term'
                query_term = "apple"

                results = collection.query(
                    query_texts=[query_term],
                    n_results=3  # Retrieve top 3 results
                )

                # Check if no results are returned or if the results array is empty
                if not results or not results['ids'] or len(results['ids'][0]) == 0:
                    # Log a message indicating that no similar documents were found for the query term
                    print(f'No documents found similar to "{query_term}"')
                    return
                
                print(f'Top 3 similar documents to "{query_term}":')
                print(f"Query results for '{query_term}':")
                print(results)

                # Access the nested arrays in 'results["ids"]' and 'results["distances"]'
                for i in range(min(3, len(results['ids'][0]))):
                    doc_id = results['ids'][0][i]  # Get ID from 'ids' array
                    score = results['distances'][0][i]  # Get score from 'distances' array
                    # Retrieve text data from the results
                    text = results['documents'][0][i]
                    if not text:
                        print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
                    else:
                        print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')
                pass
            except Exception as error:
                print(f"Error in similarity search: {error}")

        perform_similarity_search(collection, all_items)
        pass
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
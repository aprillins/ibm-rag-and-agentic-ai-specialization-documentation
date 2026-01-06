from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from llm_and_embedding import llm, openai_embedding
loader = TextLoader("companypolicies.txt")
documents = loader.load()
# print(f"Number of documents loaded: {len(documents)}")
# print(f"First document content:\n{documents[0].page_content[:500]}...")
# # print second document content if exists

# if len(documents) > 1:
#     print(f"Second document content:\n{documents[1].page_content[:500]}...")


# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# splitted_docs = splitter.split_text(documents[0])

# print(f"Number of splitted documents: {len(splitted_docs)}")

vectordb = Chroma.from_documents(documents, openai_embedding())

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

query = "What is the company's policy on remote work?"
results = retriever.invoke(query)

results



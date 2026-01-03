import chromadb
from chromadb.utils import embedding_functions
from chromadb import CloudClient
import re
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import data_loader

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

food_data = data_loader.load_food_data('FoodDataSet.json')

print(f"✅ Total food items loaded: {len(food_data)}")
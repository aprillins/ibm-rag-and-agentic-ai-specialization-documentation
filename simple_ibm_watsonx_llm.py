# Import the necessary packages
import os
from dotenv import load_dotenv
load_dotenv()

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# langchain_ibm package uses langchain-core<0.3,>=0.2.2
# If you already installed langchain-core, you can uninstall it first using:
# python -m pip uninstall langchain-core
# Then reinstall using:
# python -m pip install langchain-core==0.2.2

from langchain_ibm import WatsonxLLM

api_key = os.environ.get("WATSONX_API_KEY")

# Specify the model and project settings 
model_id = 'ibm/granite-3-8b-instruct' 

# Set the necessary parameters
params = {
    GenParams.MAX_NEW_TOKENS: 256,  # Specify the max tokens you want to generate
    GenParams.TEMPERATURE: 0.5, # This randomness or creativity of the model's responses
}

watsonx_llm = WatsonxLLM(
    model_id=model_id,
    params=params,
    url = "https://jp-tok.ml.cloud.ibm.com",
    project_id="e0eea797-c2c3-43b5-bea5-d491199d8cf7",
    apikey=api_key,
)

# Get the query from the user input
query = input("Please enter your query: ")

# Print the generated response
print(watsonx_llm.invoke(query))
import pandas as pd
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import os 
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
df = pd.read_csv(
    "student-mat.csv"
)

print(df.head())

llm = ChatOpenAI(model="gpt-4.1-mini", api_key=openai_api_key)
response = llm.invoke("are you there?")
print(f"response: {response.content}")

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=False,
    return_intermediate_steps=True,  # set return_intermediate_steps=True so that model could return code that it comes up with to generate the chart
    handle_parsing_errors=True
)

## Simple query
response = agent.invoke("how many rows of data are in this file?")
thought_process = response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n')
print(f"agent response: {response}")

## Simple query
response = agent.invoke("how many female are there in this file?")
print(f"agent response: {response['output']}")

## Generate bar chart
response = agent.invoke("Generate a bar chart to plot the gender count. Show the total number in the bar.")
print(f"agent response: {response}")

## Generate box plot
response = agent.invoke("Create box plots to analyze the relationship between 'freetime' (amount of free time) and 'G3' (final grade) across different levels of free time.")
print(f"agent response: {response}")

## Generate scatter plot
response = agent.invoke("Generate scatter plots to examine the correlation between 'Dalc' (daily alcohol consumption) and 'G3', and between 'Walc' (weekend alcohol consumption) and 'G3'.")
print(f"agent response: {response}")
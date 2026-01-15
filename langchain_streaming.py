# packages to install:
# name: data-visualization-agent
# channels:
#  - conda-forge
#  - defaults
# dependencies:
#  - python=3.11.9
#  - langchain=0.1.16
#  - langchain-openai<0.1
#  - langchain-experimental=0.0.57
#  - matplotlib=3.8.4
#  - seaborn=0.13.2
#  - tabulate
#  - python-dotenv
#  - pip

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key)

async def main():
    chain = llm | StrOutputParser()
    for chunk in chain.stream("Explain about asynchronous programming in Python in three sentences."):
        # Print the token as it arrives
        print(chunk, end="", flush=True)
    print("\n") # Add a final newline for formatting

if __name__ == "__main__":
    asyncio.run(main())

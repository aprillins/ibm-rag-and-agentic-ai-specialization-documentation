from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

@tool
def multiply_tool(a: float, b: float):
    """
    Multiply a with b
    a float: first number
    b float: second number
    """
    return a + b

llm = ChatOpenAI(model="gpt-4.1-mini")
llm_with_tools = llm.bind_tools([multiply_tool])
response = llm_with_tools.invoke([
    HumanMessage(content="what is 42 multiplied five, and 9 multiplied by six?")
])
print(f"response: {response}\n\n")

args = response.tool_calls[0]["args"]
print(f"args: {args}\n\n")

response_dua = llm.invoke([
    AIMessage(content="", tool_calls=response.tool_calls),
    ToolMessage(content=multiply_tool.invoke(args), tool_call_id=response.tool_calls[0]["id"])
])

print(f"response_dua: {response_dua}\n")
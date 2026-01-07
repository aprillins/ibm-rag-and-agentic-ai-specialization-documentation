import os
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=openai_api_key)

@tool
def calculate_tips(spending, percent_tips):
    """
        Calculate the tips that need to be given based on spending
        Args:
            spending (int): total spending
            percent_tips (float): percentage points from the total spending
        Return:
            int: spending multiplied by the percent_tips 
    """
    return spending * percent_tips

input = {
    "spending": 60,
    "percent_tips": 0.2
}

tool_map = {
    "calculate_tips": calculate_tips
}
tools = [calculate_tips]

llm_with_tools = llm.bind_tools(tools)

# query = "I have spent $60 in a restaurant. Usually the tips is 10%. How much the tip should I give?"
query = "what is cool?"
chat_history = [HumanMessage(query)]
llm_response = llm_with_tools.invoke(chat_history)
chat_history.append(llm_response)
print(f"Response: {llm_response}")


# Safely handle case where there are no tool calls
tool_calls = getattr(llm_response, "tool_calls", []) or []
if not tool_calls:
    print("No tool calls found; skipping tool invocation.")
    # If the model returned a textual response, show it as final
    if hasattr(llm_response, "content"):
        print(f"Final LLM response (no tool call): {llm_response.content}")
else:
    tool_call = tool_calls[0]
    print(f"Response: {tool_call}")

    tool_calls_id = tool_call["id"]
    tool_calls_arguments = tool_call["args"]
    tool_calls_name = tool_call["name"]

    tool_response = tool_map[tool_calls_name].invoke(tool_calls_arguments)
    tool_message = ToolMessage(content=tool_response, tool_call_id=tool_calls_id)

    chat_history.append(tool_message)

    answer = llm_with_tools.invoke(chat_history)
    print(f"Answer: {answer.content}")
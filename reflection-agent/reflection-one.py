from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
import os
from dotenv import load_dotenv
from typing import Optional
from pprint import pp
load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano")

system_msg_generate = SystemMessage("""
    You are a professional LinkedIn content assistant tasked with crafting engaging, insightful, and well-structured LinkedIn posts.
    Generate the best LinkedIn post possible for the user's request.
    If the user provides feedback or critique, respond with a refined version of your previous attempts, improving clarity, tone, or engagement as needed. 
    The max length for the post is 70 words.
    """)

system_msg_reflection = SystemMessage("""
    You are a professional LinkedIn content strategist and thought leadership expert. Your task is to critically evaluate the given LinkedIn post and provide a comprehensive critique. Follow these guidelines:

        1. Assess the post’s overall quality, professionalism, and alignment with LinkedIn best practices.
        2. Evaluate the structure, tone, clarity, and readability of the post.
        3. Analyze the post’s potential for engagement (likes, comments, shares) and its effectiveness in building professional credibility.
        4. Consider the post’s relevance to the author’s industry, audience, or current trends.
        5. Examine the use of formatting (e.g., line breaks, bullet points), hashtags, mentions, and media (if any).
        6. Evaluate the effectiveness of any call-to-action or takeaway.

        Provide a detailed critique that includes:
        - A brief explanation of the post’s strengths and weaknesses.
        - Specific areas that could be improved.
        - Actionable suggestions for enhancing clarity, engagement, and professionalism.

        Your critique will be used to improve the post in the next revision step, so ensure your feedback is thoughtful, constructive, and practical.
        Bullet points only.                              
    """)

class ProgramState(MessagesState):
    msg_count: Optional[int]

graph = StateGraph(MessagesState)

def generate_chain(state: ProgramState) -> ProgramState:
    response = llm.invoke([system_msg_generate, state["messages"][0]])
    print(f"""generate_chain len(state): {len(state["messages"])}\n""")
    return {"messages": [AIMessage(response.content)], "msg_count": 1}

def reflection_chain(state: ProgramState) -> ProgramState: 
    messages = state["messages"]
    print(f"""reflection_chain len(state): {len(state["messages"])}\n""")
    response = llm.invoke([system_msg_reflection, messages[-1]])
    # print(f"""Response in reflection_chain: \n {response.content} \n""")

    return {"messages": [HumanMessage(response.content)], "msg_count": 1}

def should_continue(state: ProgramState):
    print(f"""should_continue len(state): {len(state["messages"])}\n""")
    # print(f"""Content in should_continue: \n {state["messages"][-1]} \n""")
    if len(state["messages"]) > 4:
        return END
    return "reflection_chain"

graph.add_node("generate_chain", generate_chain)
graph.add_node("reflection_chain", reflection_chain)
graph.add_node("should_continue", should_continue)

graph.add_edge(START, "generate_chain")
graph.add_conditional_edges("generate_chain", should_continue)
graph.add_edge("reflection_chain", "generate_chain")

workflow = graph.compile()
response = workflow.invoke({"messages": [HumanMessage("create a linked in post about new crypto currency launch, in this case POLYGON")]})

for i, x in enumerate(response["messages"], start=1):
    pp(f"index {i}: {type(x).__name__}: {x.content}")
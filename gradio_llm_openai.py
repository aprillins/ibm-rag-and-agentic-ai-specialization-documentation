# Import necessary packages
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI

import gradio as gr
# Note: ChatOpenAI requires langchain-core (newest version)
# Install with: pip install langchain-openai langchain-core


# Model and project settings
model_id = 'gpt-4.1-nano' # Specify IBM's Granite 3.3 8B model
api_key=os.environ.get("OPENAI_API_KEY")
llm = ChatOpenAI(
    model_name=model_id, 
    temperature=0.5, 
    max_tokens=1000,
    api_key=api_key
    )

# Function to generate a response from the model
def generate_response(prompt_txt):
    generated_response = llm.invoke(prompt_txt)
    return generated_response.content

# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Open.ai Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)
# Requirements
# installing required libraries in my_env
# pip install transformers==4.35.2 \
# torch==2.1.1 \
# gradio==5.9.0 \
# langchain==0.3.12 \
# langchain-community==0.3.12 \
# langchain_ibm==0.3.5 \
# ibm-watsonx-ai==1.1.16 \
# pydantic==2.10.3
# Install ffmpeg
# Ubuntu
# sudo apt update
# sudo apt install ffmpeg
# Mac
# brew install ffmpeg

import torch
from transformers import pipeline
import gradio as gr

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
            )
    # Transcribe the audio file and return the result
    result = pipe(audio_file, batch_size=8)["text"]
    return result

# Set up Gradio interface
audio_input = gr.Audio(sources="upload", type="filepath")  # Audio input
output_text = gr.Textbox()  # Text output

# Create the Gradio interface with the function, inputs, and outputs
iface = gr.Interface(fn=transcript_audio, 
        inputs=audio_input, outputs=output_text, 
        title="Audio Transcription App",
        description="Upload the audio file")

# Launch the Gradio app
iface.launch(server_name="0.0.0.0", server_port=5000)
# simple Gradio interface example
# run it with `python gradio_implementation.py`

import gradio as gr

def process_text(text):
    return f"You entered: {text}"
demo = gr.Interface(fn=process_text, inputs=gr.Textbox(
    label="Enter some text", placeholder="Type here..."),
    outputs=gr.Textbox(label="Output", placeholder="Processed text will appear here..."
))
demo.launch()
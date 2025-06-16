from transformers import pipeline
import gradio as gr

model = pipeline("summarization", model="t5-large", tokenizer="t5-base")

def predict(prompt):
    formatted_input = "summarize: " + prompt.strip()
    summary = model(formatted_input, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
    return summary

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter text...", lines = 4)
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

demo.launch(share=True)

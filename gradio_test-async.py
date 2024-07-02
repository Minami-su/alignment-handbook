import re
import json
from flask import Flask, request, jsonify
from openai import OpenAI
import gradio as gr
import requests

# Gradio interface code

def generate(prompt):
    response = requests.post(
        "http://localhost:8005/predict", 
        json={"data": prompt}
    )
    return response.json()['output']

def chatbot(user_input, history):
    max_history_len = 999
    name = "<|im_start|>assistant:"
    
    # Append user input to history
    text = f"<|im_start|>user:{user_input}<|im_end|>"
    history.append([text, None])
    
    # Build the input text from the history
    input_text = ""
    for history_utr, response in history[-max_history_len:]:
        if response is not None:
            input_text += f"{history_utr}\n{response}\n"
        else:
            input_text += f"{history_utr}\n"
    
    # Prepare the prompt for the model
    prompt = input_text + name
    prompt = prompt.strip()
    print("1",prompt,"2")
    # Generate the response
    response = generate(prompt).strip()
    
    # Format the response
    formatted_response = name + response
    # Update the history with the response
    history[-1][1] = formatted_response
    
    return history, history

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 星云大模型3.5-turbo-preview-07-02")
    gr.Markdown("<small>模型还需要不断的优化迭代，所以初版效果不代表最终品质</small>")
    chatbot_output = gr.Chatbot()
    message = gr.Textbox(label="User Input")
    state = gr.State([])  # Initialize state with an empty list
    submit_button = gr.Button("Send")

    submit_button.click(chatbot, inputs=[message, state], outputs=[chatbot_output, state])

demo.launch(share=True)

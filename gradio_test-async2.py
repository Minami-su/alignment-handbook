import re
import json
from flask import Flask, request, jsonify
from openai import OpenAI
import gradio as gr
import requests
import threading

app = Flask(__name__)
system_prompt = "system:你是AI心理咨询师，由健成星云开发的，你是一个心理专家\n"#星云大模型，目前版本为：星云大模型3.5-turbo-preview-07-02
client = OpenAI(
    api_key='EMPTY',  # 如果没有设置key的话，就填 'EMPTY'
    base_url="http://localhost:8001/v1",
)

def model_predict(prompt):
    completion = client.chat.completions.create(
        model="/work/jcxy/LLaMA-Factory/model/xingyun-7B-Instruct-orpo-dialog",  # 服务器的模型名
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        top_p=0.7,
        stop="<|im_end|>",
    )

    result_content = completion.choices[0].message.content
    return result_content

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    output = model_predict(data)
    result = {'output': output}
    return jsonify(result)

# Function to run the Flask app
def run_flask():
    app.run(host='0.0.0.0', port=8005)

# Start the Flask app in a separate thread
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

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

demo.launch(share=True,)


import re
import json
from flask import Flask, request, jsonify
from openai import OpenAI
import gradio as gr
import requests

app = Flask(__name__)
system_prompt = "system:你是AI心理咨询师，由健成星云开发的星云大模型，目前版本为：星云大模型4.0-turbo-preview-07-02，你你是一个心理专家\n"
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8005)
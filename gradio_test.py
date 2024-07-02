import gradio as gr
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from transformers.generation.utils import GenerationConfig
from transformers import BitsAndBytesConfig
from peft import PeftModel

# Model and tokenizer setup
ckpt = "/work/jcxy/LLaMA-Factory/model/Qwen2-7B-Instruct"
#ckpt = "/work/jcxy/LLaMA-Factory/model/xingyun-72B-Instruct-orpo-dialog"
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    trust_remote_code=True,use_flash_attention_2=True,#attn_implementation="flash_attention_2"
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    device_map="auto"
)

model = PeftModel.from_pretrained(model, "data/Qwen2-7B-Instruct-seed_Prometheus_dpo_dialog")
device = torch.device('cuda')

# Generation function
def generate(prompt):
    print("1",prompt,"2")
    name=prompt.split("\n")[-1]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generate_ids = model.generate(
        input_ids=input_ids,
        max_length=8000,
        do_sample=True,
        top_p=0.8,
        temperature=0.7,
        top_k=20,
        repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]#
    response = output.split(name)[-1]
    return response

def chatbot(user_input, history):
    max_history_len = 999
    name = "<|im_start|>assistant:"
    
    # Append user input to history
    text = f"<|im_start|>user:{user_input}<|im_end|>"
    history.append([text, None])
    
    # Build the input text from the history
    input_text = ""
    #input_text = "<|im_start|>system:你是一个心理专家，是健成星云开发的AI心理咨询师<|im_end|>\n"
    for history_utr, response in history[-max_history_len:]:
        if response is not None:
            input_text += f"{history_utr}\n{response}\n"
        else:
            input_text += f"{history_utr}\n"
    
    
    # Prepare the prompt for the model
    prompt = input_text + name
    prompt = prompt.strip()
    
    # Generate the response
    response = generate(prompt).strip()
    
    # Format the response
    formatted_response = name + response# + "<|im_end|>"
    # Update the history with the response
    history[-1][1] = formatted_response
    
    print("1", response, "2")
    
    return history, history
# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 星云大模型4.0-turbo-preview-07-02")
    gr.Markdown("<small>模型还需要不断的优化迭代，所以初版效果不代表最终品质</small>")
    chatbot_output = gr.Chatbot()
    message = gr.Textbox(label="User Input")
    state = gr.State([])  # Initialize state with an empty list
    submit_button = gr.Button("Send")

    submit_button.click(chatbot, inputs=[message, state], outputs=[chatbot_output, state])

demo.launch(share=True)







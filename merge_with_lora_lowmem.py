import json
import os
import torch
import peft
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer
ckpt=f"/work/jcxy/LLaMA-Factory/model/Qwen2-7B-Instruct"
output_dir=f"/work/jcxy/LLaMA-Factory/model/xingyun-7B-Instruct-orpo-dialog"
tokenizer = AutoTokenizer.from_pretrained(ckpt,trust_remote_code=True)
# Original method without offloading
model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    low_cpu_mem_usage=True,
    load_in_8bit=False,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, f"data/Qwen2-7B-Instruct-seed_Prometheus_dpo_dialog",device_map={"": "cpu"},torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
tokenizer.save_pretrained(output_dir)
print("Saving to Hugging Face format...")
model.save_pretrained(output_dir,max_shard_size='2GB',
                      safe_serialization=True)
model.config.save_pretrained(output_dir)

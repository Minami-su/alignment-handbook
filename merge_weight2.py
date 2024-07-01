from safetensors import safe_open
import bitsandbytes as bnb
import torch
def find_all_linear_names(model):
    #cls = bnb.nn.Linear8bitLt 
    cls = bnb.nn.Linear4bit 
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
tensors = {}
with safe_open("/work/jcxy/haolu/workspace/alignment-handbook/data/seed_Prometheus_sft_dialog/model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k) # loads the full tensor given a key
        #print(k, tensors[k].dtype, tensors[k].shape) # Uncomment to view

from transformers import LlamaForCausalLM, BitsAndBytesConfig, AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

# Make sure the compute type, target modules, rank, alpha etc match!
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    "/work/jcxy/LLaMA-Factory/model/Qwen2-72B-Instruct",
    use_cache=False,
    quantization_config=bnb_config
)

# Freeze
for param in model.parameters():
    param.requires_grad = False
modules = find_all_linear_names(model)
print(modules)
# Add LoRA (make sure your rank (r) and alpha (lora_alpha) values match those used in training!)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=16, lora_dropout=0.1,
    target_modules=modules
)
model = get_peft_model(model, peft_config)

# Check out the first few keys in the state dict:
list(model.state_dict().keys())[:10]
new_sd = model.state_dict()
for k in new_sd:
    if 'lora' in k:
        y=k.replace("default.","")
        new_sd[k] = tensors[k]

model.load_state_dict(new_sd)

model.save_pretrained("seed_Prometheus_sft_dialog")
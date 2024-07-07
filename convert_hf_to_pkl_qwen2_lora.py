import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np
import os
import torch

model = 'qwen2_7b_lora'
lora_path = "/code/Qwen2/examples/sft/output_qwen"

# Load the model
model_configs = {
    'qwen2_7b_instruct': {
        'hf_model': "Qwen/Qwen2-7B-Instruct",
        'tokenizer': "Qwen/Qwen2-7B-Instruct",
        'weights_dir': 'weights/qwen2_7b_instruct/'
    },
    'qwen2_7b': {
        'hf_model': "Qwen/Qwen2-7B",
        'tokenizer': "Qwen/Qwen2-7B",
        'weights_dir': 'weights/qwen2_7b/'
    },
    'qwen2_7b_lora': {
        'hf_model': "Qwen/Qwen2-7B",
        'tokenizer': "Qwen/Qwen2-7B",
        'weights_dir': 'weights/qwen2_7b_lora/'
    },
}

config = model_configs[model]

# Create a directory to save the layers
os.makedirs(config['weights_dir'], exist_ok=True)

with torch.inference_mode():
    base_model = AutoModelForCausalLM.from_pretrained(config['hf_model'])
    model = PeftModel.from_pretrained(base_model, lora_path)
    # Iterate over the layers
    for w_name, layer in model.named_parameters():
        # 保存数组到 .npy 文件
        w_name = w_name.replace('base_model.model.model.', 'model.')
        w_name = w_name.replace('base_model.model.', '')
        w_name = w_name.replace('.default.', '.')
        w_name = w_name.replace('.base_layer.', '.')
        print(f'Layer {w_name}, shape {layer.shape}')
        f_name = os.path.join(config['weights_dir'], f'{w_name}.npy')
        w_tensor = layer.detach()
        # w_tensor = w_tensor.to(torch.float16)
        np.save(f_name, w_tensor.cpu().numpy())
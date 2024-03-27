import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import numpy as np
import os
import torch

model = 'llama2_7b'

# Load the model
model_configs = {
    'llama2_7b': {
        'hf_model': 'meta-llama/Llama-2-7b-hf',
        'tokenizer': 'meta-llama/Llama-2-7b-hf',
        'weights_dir': 'weights/llama2_7b/'
    },
    'llama2_70b': {
        'hf_model': 'meta-llama/Llama-2-70b-hf',
        'tokenizer': 'meta-llama/Llama-2-70b-hf',
        'weights_dir': 'weights/llama2_70b/'
    }
}

config = model_configs[model]
# tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])

# Create a directory to save the layers
os.makedirs(config['weights_dir'], exist_ok=True)

with torch.inference_mode():
    model = LlamaForCausalLM.from_pretrained(config['hf_model'])
    # Iterate over the layers
    for w_name, layer in model.named_parameters():
        # 保存数组到 .npy 文件
        print(f'Layer {w_name}, shape {layer.shape}')
        f_name = os.path.join(config['weights_dir'], f'{w_name}.npy')
        w_tensor = layer.detach()
        w_tensor = w_tensor.to(torch.float16)
        np.save(f_name, w_tensor.cpu().numpy())
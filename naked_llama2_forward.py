import os.path as osp
import torch
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
from layers.rope import init_rope_embeddings
from utils import npy_to_tensor, load_llama_config, get_attentioin_mask
from configuration_llama import LlamaConfig
from llama import LlamaModel

if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)   # 这样打印 mask 不会存在折叠的问题
    parser = argparse.ArgumentParser(description='nake.')
    parser.add_argument('--model_size', type=str, 
                        help='prammeter size of the llama2 model to use', 
                        default='7b', 
                        choices=['7b', '70b']
                        )
    args = parser.parse_args()
    
    # initial rope embeddings
    init_rope_embeddings(dim=128)
    prompt = "Hey, are you conscious? Can you talk to me?"
    model_dict = {
        "llama2_7b": {
            'tokenizer': 'meta-llama/Llama-2-7b-hf',
            'hf_model': 'meta-llama/Llama-2-7b-hf',
            'config_path': 'configs/llama2_7b_config.json',
            'weights_dir': 'weights/llama2_7b/'
        },
        "llama2_70b": {
            'tokenizer': 'meta-llama/Llama-2-70b-hf',
            'hf_model': 'meta-llama/Llama-2-70b-hf',
            'config_path': 'configs/llama2_70b_config.json',
            'weights_dir': 'weights/llama2_70b/'
        }
    }
    if args.model_size == '7b':
        model_name = "llama2_7b"
    elif args.model_size == '70b':
        model_name = "llama2_70b"
        
    print('Model:', model_name)   
    
    # tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name]['tokenizer'])
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids

    # random input
    # token_ids = torch.randint(0, 32000, (1, 512))  # (1, 512) shape

    config = load_llama_config(model_dict[model_name]['config_path'])
    config.weights_dir = model_dict[model_name]['weights_dir']
    llama2 = LlamaModel(config)
    logits = llama2.forward(token_ids)
    
    print(f'Naked llama, model: {config.model_name}, result:')
    print(logits)
    
    # check result
    model = LlamaForCausalLM.from_pretrained(model_dict[model_name]['hf_model'])
    model.eval()
    with torch.inference_mode():
        hf_res = model(input_ids = token_ids)
        print(f'Hugging face, model: {config.model_name}, result:')
        print(hf_res.logits)
    error = torch.abs(hf_res.logits-logits)
    print(f"Compare error sum: {torch.sum(error)}") 

    
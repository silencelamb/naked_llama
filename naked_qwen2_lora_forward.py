import os.path as osp
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from layers.rope import init_rope_embeddings
from utils import npy_to_tensor, load_llama_config, get_attentioin_mask
from configuration_llama import LlamaConfig
from llama import LoraLlamaModel

if __name__ == '__main__':
    torch.set_printoptions(linewidth=200)   # 这样打印 mask 不会存在折叠的问题
    parser = argparse.ArgumentParser(description='nake.')
    parser.add_argument('--model_size', type=str, 
                        help='prammeter size of the llama2 model to use', 
                        default='7b', 
                        choices=['7b', '70b']
                        )
    args = parser.parse_args()
    

    prompt = "Hey, are you conscious? Can you talk to me?"
    model_dict = {
        "qwen2_7b_instruct": {
            'tokenizer': "Qwen/Qwen2-7B-Instruct",
            'hf_model': "Qwen/Qwen2-7B-Instruct",
            'config_path': 'configs/qwen2_7b_instruct_config.json',
            'weights_dir': 'weights/qwen2_7b_instruct/'
        },
        "qwen2_7b": {
            'tokenizer': "Qwen/Qwen2-7B",
            'hf_model': "Qwen/Qwen2-7B",
            'config_path': 'configs/qwen2_7b_config.json',
            'weights_dir': 'weights/qwen2_7b/'
        },
        "qwen2_7b_lora": {
            'tokenizer': "Qwen/Qwen2-7B-Instruct",
            'hf_model': "Qwen/Qwen2-7B-Instruct",
            'config_path': 'configs/qwen2_7b_instruct_config.json',
            'weights_dir': 'weights/qwen2_7b_lora/'
        },
    }
    if args.model_size == '7b':
        model_name = "qwen2_7b_lora"
        
    print('Model:', model_name)
    
    # tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name]['tokenizer'])
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids

    # random input
    # token_ids = torch.randint(0, 32000, (1, 512))  # (1, 512) shape

    config = load_llama_config(model_dict[model_name]['config_path'])
    config.weights_dir = model_dict[model_name]['weights_dir']
    
    # initial rope embeddings
    init_rope_embeddings(dim=128, max_position_embeddings=config.max_position_embeddings, base=config.rope_theta)
    
    llama2 = LoraLlamaModel(config)
    print('loaded base weights!')
    lora_alpha = 16
    lora_r = 64
    llama2.replace_with_lora(config, scaling=lora_alpha/lora_r, dropout=0.05)
    llama2.eval()
    print('loaded lora weights!')
    logits = llama2.forward(token_ids)
    
    print(f'Naked llama, model: {config.model_name}, result:')
    print(logits)
    
    # check result
    base_model = AutoModelForCausalLM.from_pretrained(model_dict[model_name]['hf_model'], torch_dtype=torch.float32)
    lora_path = "/code/Qwen2/examples/sft/output_qwen"
    model = PeftModel.from_pretrained(base_model, lora_path, torch_dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        hf_res = model(input_ids = token_ids)
        print(f'Hugging face, model: {config.model_name}, result:')
        print(hf_res.logits)
    error = torch.abs(hf_res.logits-logits)
    print(f"Compare error sum: {torch.sum(error)}") 
    print("torch.testing.assert_close", torch.testing.assert_close(hf_res.logits, logits))

    
import os.path as osp
import torch
import argparse
from transformers import AutoTokenizer, LlamaForCausalLM
from utils import npy_to_tensor, load_llama_config, get_attentioin_mask
from configuration_llama import LlamaConfig
from layers.norm import RMSNorm
from layers.rope import init_rope_embeddings
from layers.embedding import embedding_lookup
from layers.matmul import LlamaMLP, lm_head
from layers.transformer_block import llama2_transformer_block


def llama2(token_ids: torch.Tensor, config: LlamaConfig):
    """
    手动实现llama2 7B/13B/70B的推理计算。
    
    参数:
    - token_ids: token id组成的tensor，形状为 [batch_size, seq_length]
    """
    bsz, seq_length = token_ids.shape
    # embedding 
    embdding_weights = npy_to_tensor(osp.join(config.weights_dir, 'model.embed_tokens.weight.npy'))
    input_embeds = embedding_lookup(token_ids, embdding_weights)
    hidden_states = input_embeds  # shape [batch_size, seq_length, hidden_size], hidden_size=4096
    
    # mask
    mask = get_attentioin_mask(start_pos=0, seq_length=seq_length, ref_tensor=hidden_states)

    # 重复 32次(7B)/ 80次(70B) llama2_transformer_block 的计算
    for layer_id in range(config.num_hidden_layers):
        print(f'Naked llama2: Computing Layer {layer_id}')
        output = llama2_transformer_block(hidden_states, config, layer_id=layer_id, attention_mask=mask)
        hidden_states = output[0]
    
    # 先 RMSNorm，然后head输出
    norm_weight = npy_to_tensor(osp.join(config.weights_dir, 'model.norm.weight.npy'))
    hidden_states = RMSNorm(hidden_states, norm_weight, eps=config.rms_norm_eps)
    
    lm_head_weight = npy_to_tensor(osp.join(config.weights_dir, 'lm_head.weight.npy'))
    logits = lm_head(hidden_states, lm_head_weight)
    return logits


if __name__ == '__main__':
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
            'config_path': 'configs/llama2_7b_config.json',
            'weights_dir': 'weights/llama2_7b/'
        },
        "llama2_70b": {
            'tokenizer': 'meta-llama/Llama-2-70b-hf',
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
    logits = llama2(token_ids, config)
    
    print('Naked llama result:')
    print(logits)
    
    # check result
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    model.eval()
    with torch.inference_mode():
        hf_res = model(input_ids = token_ids)
        print('Hugging face llama result:')
        print(hf_res.logits)
    error = torch.abs(hf_res.logits-logits)
    print(f"Compare error sum: {torch.sum(error)}") 

    
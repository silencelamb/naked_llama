import os.path as osp
import torch
from torch import nn
from layers.embedding import Embedding
from layers.transformer_block import LlamaTransformerBlock
from layers.rms_norm import LlamaRMSNorm
from layers.matmul import MLP
from layers.rope import init_rope_embeddings
from utils import npy_to_tensor, load_llama_config, get_attentioin_mask
from configuration_llama import LlamaConfig

class LlamaModel():
    def __init__(self, config: LlamaConfig):
        self.config = config
        embdding_weights = npy_to_tensor(osp.join(config.weights_dir, 'model.embed_tokens.weight.npy'))
        self.embdding = Embedding(embdding_weights)
        
        self.transformer_blocks = []
        for i in range(config.num_hidden_layers):
            w_q = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.weight.npy'))
            w_k = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.weight.npy'))
            w_v = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.weight.npy'))
            w_o = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.weight.npy'))
            input_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.input_layernorm.weight.npy'))
            post_att_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.post_attention_layernorm.weight.npy'))
            w_up = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.up_proj.weight.npy'))
            w_gate = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.gate_proj.weight.npy'))
            w_down = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.down_proj.weight.npy'))
            bias_q_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.bias.npy')
            bias_q = npy_to_tensor(bias_q_file) if osp.exists(bias_q_file) else None
            bias_k_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.bias.npy')
            bias_k = npy_to_tensor(bias_k_file) if osp.exists(bias_k_file) else None
            bias_v_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.bias.npy')
            bias_v = npy_to_tensor(bias_v_file) if osp.exists(bias_v_file) else None
            bias_o_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.bias.npy')
            bias_o = npy_to_tensor(bias_o_file) if osp.exists(bias_o_file) else None
            transformer_block = LlamaTransformerBlock(config, input_norm_weight, w_q, w_k, w_v, w_o, w_up, w_gate, w_down, \
                post_att_norm_weight, bias_q, bias_k, bias_v, bias_o)
            self.transformer_blocks.append(transformer_block)
        
        norm_weight = npy_to_tensor(osp.join(config.weights_dir, 'model.norm.weight.npy'))
        self.output_rmsnorm = LlamaRMSNorm(norm_weight, config.rms_norm_eps)
        
        lm_head_weight = npy_to_tensor(osp.join(config.weights_dir, 'lm_head.weight.npy'))
        self.lm_head = MLP(lm_head_weight)
        
        self.cache = None
    
    def forward(self, token_ids: torch.Tensor):
        """
        手动实现llama2 7B/13B/70B的推理计算。
        
        参数:
        - token_ids: token id组成的tensor，形状为 [batch_size, seq_length]
        """
        bsz, seq_length = token_ids.shape
        # embedding 
        
        input_embeds = self.embdding.forward(token_ids)
        hidden_states = input_embeds  # shape [batch_size, seq_length, hidden_size], hidden_size=4096
        
        # mask
        mask = get_attentioin_mask(start_pos=0, seq_length=seq_length, ref_tensor=hidden_states)

        # 重复 32次(7B)/ 80次(70B) llama2_transformer_block 的计算
        
        for layer_id, transformer_block in enumerate(self.transformer_blocks):
            print(f'Naked llama: Computing {self.config.model_name} Layer {layer_id}')
            hidden_states = transformer_block.forward(hidden_states, attention_mask=mask)
        
        # 先 RMSNorm，然后head输出
        hidden_states = self.output_rmsnorm.forward(hidden_states)
        logits = self.lm_head.forward(hidden_states)        

        return logits
    
    def backward(self, grad_output):
        pass



def test_llama_backward_manual_class():

    pass

if __name__ == '__main__':

    test_llama_backward_manual_class()
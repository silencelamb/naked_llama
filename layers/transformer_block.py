import torch
from .attention import multi_head_attention
from .norm import RMSNorm
from .matmul import LlamaMLP
from utils import npy_to_tensor

def llama2_transformer_block(hidden_states, num_heads, layer_id, attention_mask=None, use_cache=False, present_key_value=None):
    
    
    w_q = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.self_attn.q_proj.weight.npy')
    w_k = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.self_attn.k_proj.weight.npy')
    w_v = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.self_attn.v_proj.weight.npy')
    w_o = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.self_attn.o_proj.weight.npy')
    
    residual = hidden_states
    # input RMS Norm
    input_norm_weight = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.input_layernorm.weight.npy')
    hidden_states = RMSNorm(hidden_states, weight=input_norm_weight, eps=1e-5)
    
    # 多头自注意力层
    
    hidden_states = multi_head_attention(hidden_states, w_q, w_k, w_v, w_o, num_heads, attention_mask)
    
    # 残差连接
    hidden_states = residual + hidden_states  

    # FFN 计算部分
    residual = hidden_states
    # post attention RMS Norm
    post_att_norm_weight = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.post_attention_layernorm.weight.npy')
    hidden_states = RMSNorm(hidden_states, weight=post_att_norm_weight, eps=1e-5)
    
    # FFN up & FFN gate & FFN down
    w_up = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.mlp.up_proj.weight.npy')
    w_gate = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.mlp.gate_proj.weight.npy')
    w_down = npy_to_tensor(f'weights/llama2_7b/model.layers.{layer_id}.mlp.down_proj.weight.npy')
    hidden_states = LlamaMLP(hidden_states, w_up, w_gate, w_down)
    
    
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    # 如果cache key和value，则将它们添加到输出中
    if use_cache:
        outputs += (present_key_value,)

    return outputs
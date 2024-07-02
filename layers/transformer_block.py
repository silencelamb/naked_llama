import os.path as osp
import torch
from layers.attention import multi_head_attention, multi_head_attention_backward
from layers.rms_norm import RMSNorm, RMSNorm_backward
from layers.matmul import LlamaMLP, LlamaMLP_backward
from utils import npy_to_tensor, load_llama_config
from configuration_llama import LlamaConfig

def llama2_transformer_block(hidden_states,
                             config: LlamaConfig, 
                             layer_id, 
                             attention_mask=None, 
                             use_cache=False, 
                             present_key_value=None):

    w_q = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.q_proj.weight.npy'))
    w_k = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.k_proj.weight.npy'))
    w_v = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.v_proj.weight.npy'))
    w_o = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.o_proj.weight.npy'))

    w_q.requires_grad_(True)
    w_k.requires_grad_(True)
    w_v.requires_grad_(True)
    w_o.requires_grad_(True)
    
    residual = hidden_states
    # input RMS Norm
    input_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.input_layernorm.weight.npy'))
    input_norm_weight.requires_grad_(True)
    hidden_states = RMSNorm(hidden_states, weight=input_norm_weight, eps=config.rms_norm_eps)
    
    # 多头自注意力层
    
    hidden_states = multi_head_attention(hidden_states, w_q, w_k, w_v, w_o, config, attention_mask)
    
    # 残差连接
    hidden_states = residual + hidden_states  

    # FFN 计算部分
    residual = hidden_states
    # post attention RMS Norm
    post_att_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.post_attention_layernorm.weight.npy'))
    post_att_norm_weight.requires_grad_(True)
    hidden_states = RMSNorm(hidden_states, weight=post_att_norm_weight, eps=config.rms_norm_eps)
    
    # FFN up & FFN gate & FFN down
    w_up = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.up_proj.weight.npy'))
    w_gate = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.gate_proj.weight.npy'))
    w_down = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.down_proj.weight.npy'))
    w_up.requires_grad_(True)
    w_gate.requires_grad_(True)
    w_down.requires_grad_(True)

    hidden_states = LlamaMLP(hidden_states, w_up, w_gate, w_down)
    
    
    hidden_states = residual + hidden_states

    outputs = (hidden_states, w_q, w_k, w_v, w_o, input_norm_weight, post_att_norm_weight, w_up, w_gate, w_down)

    # 如果cache key和value，则将它们添加到输出中
    if use_cache:
        outputs += (present_key_value,)

    return outputs

def llama2_transformer_block_backward(grad_output, 
                                      hidden_states, 
                                      config: LlamaConfig, 
                                      layer_id, 
                                      attention_mask=None,
                                      use_cache=False, 
                                      present_key_value=None):
    # 提取权重
    w_q = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.q_proj.weight.npy'))
    w_k = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.k_proj.weight.npy'))
    w_v = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.v_proj.weight.npy'))
    w_o = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.o_proj.weight.npy'))
    
    w_q.requires_grad_(True)
    w_k.requires_grad_(True)
    w_v.requires_grad_(True)
    w_o.requires_grad_(True)
    
    input_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.input_layernorm.weight.npy'))
    post_att_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.post_attention_layernorm.weight.npy'))
    w_up = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.up_proj.weight.npy'))
    w_gate = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.gate_proj.weight.npy'))
    w_down = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.down_proj.weight.npy'))

    input_norm_weight.requires_grad_(True)
    post_att_norm_weight.requires_grad_(True)
    w_up.requires_grad_(True)
    w_gate.requires_grad_(True)
    w_down.requires_grad_(True)

    # Forward pass to get all intermediate values
    residual_1 = hidden_states
    hidden_states_1 = RMSNorm(hidden_states, weight=input_norm_weight, eps=config.rms_norm_eps)
    hidden_states_2 = multi_head_attention(hidden_states_1, w_q, w_k, w_v, w_o, config, attention_mask)
    hidden_states_3 = residual_1 + hidden_states_2
    residual_2 = hidden_states_3
    hidden_states_4 = RMSNorm(hidden_states_3, weight=post_att_norm_weight, eps=config.rms_norm_eps)
    hidden_states_5 = LlamaMLP(hidden_states_4, w_up, w_gate, w_down)
    hidden_states_6 = residual_2 + hidden_states_5

    # 第二次残差反向梯度
    grad_residual_2 = grad_output
    grad_hidden_states_5 = grad_output
    # LlamaMLP 反向梯度 
    grad_hidden_states_4, grad_w_up, grad_w_gate, grad_w_down = LlamaMLP_backward(grad_hidden_states_5, hidden_states_4, w_up, w_gate, w_down)
    # pos_att RMSNorm 反向梯度
    grad_hidden_states_3_post, grad_post_att_norm_weight = RMSNorm_backward(grad_hidden_states_4, hidden_states_3, post_att_norm_weight)
    
    # 第一次残差反向梯度
    grad_hidden_states_3 = grad_hidden_states_3_post + grad_output
    grad_residual_1 = grad_hidden_states_3
    grad_hidden_states_2 = grad_hidden_states_3

    # Multi_head_attention 反向梯度
    grad_hidden_states_1, grad_w_q, grad_w_k, grad_w_v, grad_w_o = multi_head_attention_backward(grad_hidden_states_2, hidden_states_1, w_q, w_k, w_v, w_o, config, attention_mask)

    # input RMSNorm 反向梯度
    grad_hidden_states, grad_input_norm_weight = RMSNorm_backward(grad_hidden_states_1, hidden_states, input_norm_weight)

    grad_hidden_states += grad_residual_1  # Include residual connection

    grads = (grad_hidden_states, grad_input_norm_weight, grad_w_q, grad_w_k, grad_w_v, grad_w_o, \
              grad_post_att_norm_weight, grad_w_up, grad_w_gate, grad_w_down)

    return grads

def test_transformer_block_backward_manual_func():

    model_dict = {
        "llama2_7b": {
            'tokenizer': 'meta-llama/Llama-2-7b-hf',
            'hf_model': 'meta-llama/Llama-2-7b-hf',
            'config_path': '/code/naked_llama/configs/llama2_7b_config.json',
            'weights_dir': '/code/naked_llama/weights/llama2_7b/'
        },}
    model_name = "llama2_7b"
    config = load_llama_config(model_dict[model_name]['config_path'])
    config.weights_dir = model_dict[model_name]['weights_dir']

    batch_size, seq_len, hidden_size = 1, 6, 4096
    from layers.rope import init_rope_embeddings
    init_rope_embeddings(dim=128)
    layer_id = 1    
    x = torch.randn(batch_size, seq_len, hidden_size) 
    x.requires_grad_(True)
    outputs = llama2_transformer_block(x, config, layer_id=layer_id)
    output, w_q, w_k, w_v, w_o, input_norm_weight, post_att_norm_weight, w_up, w_gate, w_down = outputs
    grad_output = .1 * torch.randn_like(output)
    output.backward(grad_output, retain_graph=True) 

    auto_grads = [_.grad.clone() for _ in [x, input_norm_weight, w_q, w_k, w_v, w_o,  
                                         post_att_norm_weight, w_up, w_gate, w_down]]
                 
    x = x.clone().detach().requires_grad_(True)
    manual_grads = llama2_transformer_block_backward(grad_output, x, config, layer_id=layer_id)
    gradient_names = ["x", "input_norm_weight", "w_q", "w_k", "w_v", "w_o", "post_att_norm_weight", "w_up", "w_gate", "w_down"]
    for name, auto_grad, manual_grad in zip(gradient_names, auto_grads, manual_grads):
        try:
            torch.testing.assert_close(auto_grad, manual_grad)
            print(f"Gradient match for {name}: PASSED")
        except AssertionError as e:
            print(f"Gradient match for {name}: FAILED")
            print(e)

if __name__ == '__main__':

    test_transformer_block_backward_manual_func()
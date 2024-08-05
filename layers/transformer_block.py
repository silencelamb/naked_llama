import os.path as osp
import torch
from torch import nn
from layers.attention import LlamaAttention, LoraLlamaAttention
from layers.rms_norm import LlamaRMSNorm
from layers.matmul import LlamaMLP, LoraLlamaMLP
from layers.rope import init_rope_embeddings
from utils import npy_to_tensor, load_llama_config, get_attentioin_mask
from configuration_llama import LlamaConfig

class LlamaTransformerBlock():
    def __init__(self, config: LlamaConfig, input_norm_weight, w_q, w_k, w_v, w_o, 
                 w_up, w_gate, w_down, post_att_norm_weight,
                 bias_q=None, bias_k=None, bias_v=None, bias_o=None,
                 bias_up=None, bias_gate=None, bias_down=None):
        self.input_rmsnrom = LlamaRMSNorm(input_norm_weight, eps=config.rms_norm_eps)
        self.llama_attention = LlamaAttention(config, w_q, w_k, w_v, w_o, bias_q, bias_k, bias_v, bias_o)
        self.post_att_rmsnorm = LlamaRMSNorm(post_att_norm_weight, eps=config.rms_norm_eps)
        self.llama_mlp = LlamaMLP(w_up, w_gate, w_down, bias_up, bias_gate, bias_down)
        self.cache = None
    
    def forward(self, hidden_states, attention_mask, use_cache=False):
        residual_1 = hidden_states
        # input RMS Norm
        hidden_states_1 = self.input_rmsnrom.forward(hidden_states)
        # 多头自注意力层
        
        hidden_states_2 = self.llama_attention.forward(hidden_states_1, attention_mask)
        
        # 残差连接
        hidden_states_3 = residual_1 + hidden_states_2  

        # FFN 计算部分
        residual_2 = hidden_states_3
        # post attention RMS Norm
        hidden_states_4 = self.post_att_rmsnorm.forward(hidden_states_3)
        
        # FFN up & FFN gate & FFN down
        hidden_states_5 = self.llama_mlp.forward(hidden_states_4)

        hidden_states_6 = residual_2 + hidden_states_5
        
        outputs = hidden_states_6

        return outputs
    
    def backward(self, grad_output):
        # 第二次残差反向梯度
        # LlamaMLP 反向梯度 
        grad_hidden_states_4, grad_w_up, grad_w_gate, grad_w_down, _, _, _ = self.llama_mlp.backward(grad_output)
        # pos_att RMSNorm 反向梯度
        grad_hidden_states_3_post, grad_post_att_norm_weight = self.post_att_rmsnorm.backward(grad_hidden_states_4)
        
        # 第一次残差反向梯度
        grad_hidden_states_3 = grad_hidden_states_3_post + grad_output
        grad_residual_1 = grad_hidden_states_3
        grad_hidden_states_2 = grad_hidden_states_3

        # Multi_head_attention 反向梯度
        grad_hidden_states_1, grad_w_q, grad_w_k, grad_w_v, grad_w_o, grad_bias_q, grad_bias_k, grad_bias_v,\
            grad_bias_o = self.llama_attention.backward(grad_hidden_states_2)

        # input RMSNorm 反向梯度
        grad_hidden_states, grad_input_norm_weight = self.input_rmsnrom.backward(grad_hidden_states_1)

        grad_hidden_states += grad_residual_1  # Include residual connection

        grads = (grad_hidden_states, grad_input_norm_weight, grad_w_q, grad_w_k, grad_w_v, grad_w_o, \
            grad_post_att_norm_weight, grad_w_up, grad_w_gate, grad_w_down, grad_bias_q, grad_bias_k, \
                    grad_bias_v, grad_bias_o)

        return grads

class LoraLlamaTransformerBlock(LlamaTransformerBlock):
    def __init__(self, config: LlamaConfig, input_norm_weight, w_q, w_k, w_v, w_o, 
                 w_up, w_gate, w_down, post_att_norm_weight,
                 bias_q=None, bias_k=None, bias_v=None, bias_o=None,
                 bias_up=None, bias_gate=None, bias_down=None):
        super(LoraLlamaTransformerBlock, self).__init__(config, input_norm_weight, w_q, w_k, w_v, w_o, w_up, w_gate, \
            w_down, post_att_norm_weight, bias_q, bias_k, bias_v, bias_o, bias_up, bias_gate, bias_down)
        self.llama_attention = LoraLlamaAttention(config, w_q, w_k, w_v, w_o, bias_q, bias_k, bias_v, bias_o)
        self.llama_mlp = LoraLlamaMLP(w_up, w_gate, w_down, bias_up, bias_gate, bias_down)
    
    def replace_with_lora(self, q_lora_a, q_lora_b, k_lora_a, k_lora_b, v_lora_a, v_lora_b, o_lora_a, o_lora_b, \
        up_lora_a, up_lora_b, gate_lora_a, gate_lora_b, down_lora_a, down_lora_b, scaling, dropout):
        self.llama_attention.replace_with_lora(q_lora_a, q_lora_b, k_lora_a, k_lora_b, v_lora_a, v_lora_b, o_lora_a, \
            o_lora_b, scaling, dropout)
        self.llama_mlp.replace_with_lora(up_lora_a, up_lora_b, gate_lora_a, gate_lora_b, down_lora_a, down_lora_b, \
            scaling, dropout)

    def eval(self):
        self.llama_attention.eval()
        self.llama_mlp.eval()
    
    def train(self):
        self.llama_attention.train()
        self.llama_mlp.train()
        
    def backward(self, grad_output):
        # 第二次残差反向梯度
        # LlamaMLP 反向梯度 
        grad_hidden_states_4, grad_lora_a_up, grad_lora_b_up, grad_lora_a_gate, \
            grad_lora_b_gate, grad_lora_a_down, grad_lora_b_down = self.llama_mlp.backward(grad_output)
        # pos_att RMSNorm 反向梯度
        grad_hidden_states_3_post, _ = self.post_att_rmsnorm.backward(grad_hidden_states_4)
        
        # 第一次残差反向梯度
        grad_hidden_states_3 = grad_hidden_states_3_post + grad_output
        grad_residual_1 = grad_hidden_states_3
        grad_hidden_states_2 = grad_hidden_states_3

        # Multi_head_attention 反向梯度
        grad_hidden_states_1, grad_lora_a_q, grad_lora_b_q, grad_lora_a_k, grad_lora_b_k, \
            grad_lora_a_v, grad_lora_b_v, grad_lora_a_o, grad_lora_b_o \
                  = self.llama_attention.backward(grad_hidden_states_2)

        # input RMSNorm 反向梯度
        grad_hidden_states, _ = self.input_rmsnrom.backward(grad_hidden_states_1)

        grad_hidden_states += grad_residual_1  # Include residual connection

        grads = (grad_hidden_states, grad_lora_a_q, grad_lora_b_q, grad_lora_a_k, \
                  grad_lora_b_k, grad_lora_a_v, grad_lora_b_v, grad_lora_a_o, grad_lora_b_o, \
                    grad_lora_a_up, grad_lora_b_up, grad_lora_a_gate, \
                     grad_lora_b_gate, grad_lora_a_down, grad_lora_b_down)

        return grads

def test_transformer_block_backward_manual_class():

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
    num_heads, num_kv_heads = 32, 32
    head_dim = hidden_size // num_heads
    intermediate_size = 11008
    
    init_rope_embeddings(dim=128)
    layer_id = 1    
    x = torch.randn(batch_size, seq_len, hidden_size).requires_grad_(True)
    
    input_norm_weight = torch.randn(hidden_size).requires_grad_(True)
    # w_q = torch.randn(num_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    # w_k = torch.randn(num_kv_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    # w_v = torch.randn(num_kv_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    # w_o = torch.randn(num_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    post_att_norm_weight = torch.randn(hidden_size).requires_grad_(True)
    # w_up = torch.randn(intermediate_size, hidden_size).requires_grad_(True)
    # w_gate = torch.randn(intermediate_size, hidden_size).requires_grad_(True)
    # w_down = torch.randn(hidden_size, intermediate_size).requires_grad_(True)
    
    w_q = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.q_proj.weight.npy'))
    w_k = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.k_proj.weight.npy'))
    w_v = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.v_proj.weight.npy'))
    w_o = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.o_proj.weight.npy'))
    # input_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.input_layernorm.weight.npy'))
    # post_att_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.post_attention_layernorm.weight.npy'))
    w_up = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.up_proj.weight.npy'))
    w_gate = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.gate_proj.weight.npy'))
    w_down = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.down_proj.weight.npy'))

    bias_q = torch.randn(num_heads*head_dim).requires_grad_(True)
    bias_k = torch.randn(num_kv_heads*head_dim).requires_grad_(True)
    bias_v = torch.randn(num_kv_heads*head_dim).requires_grad_(True)
    bias_o = torch.randn(num_heads*head_dim).requires_grad_(True)
    
    w_q.requires_grad_(True)
    w_k.requires_grad_(True)
    w_v.requires_grad_(True)
    w_o.requires_grad_(True)
    w_up.requires_grad_(True)
    w_gate.requires_grad_(True)
    w_down.requires_grad_(True)
    input_norm_weight.requires_grad_(True)
    post_att_norm_weight.requires_grad_(True)
    
    llama2_transformer_block = LlamaTransformerBlock(config, input_norm_weight, w_q, w_k, w_v, w_o, w_up, w_gate, w_down, \
        post_att_norm_weight, bias_q, bias_k, bias_v, bias_o)
    mask = get_attentioin_mask(start_pos=0, seq_length=seq_len, ref_tensor=x)
    output = llama2_transformer_block.forward(x, attention_mask=mask)
    grad_output = .1 * torch.randn_like(output)
    output.backward(grad_output, retain_graph=True) 
    # print(output)

    auto_grads = [_.grad.clone() for _ in [x, input_norm_weight, w_q, w_k, w_v, w_o, \
        post_att_norm_weight, w_up, w_gate, w_down, bias_q, bias_k, bias_v, bias_o]]

    x = x.clone().detach().requires_grad_(True)
    manual_grads = llama2_transformer_block.backward(grad_output)
    gradient_names = ["x", "input_norm_weight", "w_q", "w_k", "w_v", "w_o", "post_att_norm_weight", \
        "w_up", "w_gate", "w_down", "bias_q", "bias_k", "bias_v", "bias_o"]
    for name, auto_grad, manual_grad in zip(gradient_names, auto_grads, manual_grads):
        try:
            torch.testing.assert_close(auto_grad, manual_grad)
            print(f"Gradient match for {name}: PASSED")
        except AssertionError as e:
            print(f"Gradient match for {name}: FAILED")
            print(e)

def test_lora_transformer_block_backward_manual_class():

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
    num_heads, num_kv_heads = 32, 32
    head_dim = hidden_size // num_heads
    intermediate_size = 11008
    rank = 16
    dropout_rate = 0.1
    scaling_factor = 0.6
    
    init_rope_embeddings(dim=128)
    layer_id = 1    
    x = torch.randn(batch_size, seq_len, hidden_size).requires_grad_(True)
    
    input_norm_weight = torch.randn(hidden_size).requires_grad_(True)
    # w_q = torch.randn(num_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    # w_k = torch.randn(num_kv_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    # w_v = torch.randn(num_kv_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    # w_o = torch.randn(num_heads*head_dim, num_heads*head_dim).requires_grad_(True)
    post_att_norm_weight = torch.randn(hidden_size).requires_grad_(True)
    # w_up = torch.randn(intermediate_size, hidden_size).requires_grad_(True)
    # w_gate = torch.randn(intermediate_size, hidden_size).requires_grad_(True)
    # w_down = torch.randn(hidden_size, intermediate_size).requires_grad_(True)
    
    w_q = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.q_proj.weight.npy'))
    w_k = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.k_proj.weight.npy'))
    w_v = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.v_proj.weight.npy'))
    w_o = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.self_attn.o_proj.weight.npy'))
    # input_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.input_layernorm.weight.npy'))
    # post_att_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.post_attention_layernorm.weight.npy'))
    w_up = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.up_proj.weight.npy'))
    w_gate = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.gate_proj.weight.npy'))
    w_down = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{layer_id}.mlp.down_proj.weight.npy'))

    bias_q = torch.randn(num_heads*head_dim).requires_grad_(True)
    bias_k = torch.randn(num_kv_heads*head_dim).requires_grad_(True)
    bias_v = torch.randn(num_kv_heads*head_dim).requires_grad_(True)
    bias_o = torch.randn(num_heads*head_dim).requires_grad_(True)
    
    w_q.requires_grad_(True)
    w_k.requires_grad_(True)
    w_v.requires_grad_(True)
    w_o.requires_grad_(True)
    w_up.requires_grad_(True)
    w_gate.requires_grad_(True)
    w_down.requires_grad_(True)
    input_norm_weight.requires_grad_(True)
    post_att_norm_weight.requires_grad_(True)

    std_dev = 0.01

    lora_a_q = torch.randn(rank, num_heads*head_dim)*std_dev
    lora_b_q = torch.randn(num_heads*head_dim, rank)*std_dev
    lora_a_k = torch.randn(rank, num_heads*head_dim)*std_dev
    lora_b_k = torch.randn(num_heads*head_dim, rank)*std_dev
    lora_a_v = torch.randn(rank, num_heads*head_dim)*std_dev
    lora_b_v = torch.randn(num_heads*head_dim, rank)*std_dev
    lora_a_o = torch.randn(rank, num_heads*head_dim)*std_dev
    lora_b_o = torch.randn(num_heads*head_dim, rank)*std_dev

    lora_a_q.requires_grad_(True)
    lora_b_q.requires_grad_(True)
    lora_a_k.requires_grad_(True)
    lora_b_k.requires_grad_(True)
    lora_a_v.requires_grad_(True)
    lora_b_v.requires_grad_(True)
    lora_a_o.requires_grad_(True)
    lora_b_o.requires_grad_(True)

    up_lora_a = torch.randn(rank, hidden_size)*std_dev
    up_lora_b = torch.randn(intermediate_size, rank)*std_dev
    gate_lora_a = torch.randn(rank, hidden_size)*std_dev
    gate_lora_b = torch.randn(intermediate_size, rank)*std_dev
    down_lora_a = torch.randn(rank, intermediate_size)*std_dev
    down_lora_b = torch.randn(hidden_size, rank)*std_dev

    up_lora_a.requires_grad_(True)
    up_lora_b.requires_grad_(True)
    gate_lora_a.requires_grad_(True)
    gate_lora_b.requires_grad_(True)
    down_lora_a.requires_grad_(True)
    down_lora_b.requires_grad_(True)

    lora_llama2_transformer_block = LoraLlamaTransformerBlock(config, input_norm_weight, w_q, w_k, w_v, w_o, w_up, w_gate, w_down, \
        post_att_norm_weight, bias_q, bias_k, bias_v, bias_o)
    lora_llama2_transformer_block.replace_with_lora(lora_a_q,lora_b_q,lora_a_k,lora_b_k,lora_a_v,\
                                                lora_b_v,lora_a_o,lora_b_o, up_lora_a, up_lora_b, gate_lora_a, \
                                                    gate_lora_b, down_lora_a, down_lora_b, scaling_factor, dropout_rate)
    mask = get_attentioin_mask(start_pos=0, seq_length=seq_len, ref_tensor=x)
    output = lora_llama2_transformer_block.forward(x, attention_mask=mask)
    grad_output = .1 * torch.randn_like(output)
    output.backward(grad_output, retain_graph=True) 

    auto_grads = [_.grad.clone() for _ in [x, lora_a_q, lora_b_q, lora_a_k, \
                                            lora_b_k, lora_a_v, lora_b_v, lora_a_o, lora_b_o, \
                                            up_lora_a, up_lora_b, gate_lora_a, \
                                            gate_lora_b, down_lora_a, down_lora_b]]

    x = x.clone().detach().requires_grad_(True)
    manual_grads = lora_llama2_transformer_block.backward(grad_output)
    # print(lora_a_q.grad.clone())
    # print(manual_grads[1])
    gradient_names = ["x", "lora_a_q", "lora_b_q", "lora_a_k", \
                           "lora_b_k", "lora_a_v", "lora_b_v", "lora_a_o", "lora_b_o", \
                                "up_lora_a", "up_lora_b", "gate_lora_a", \
                                    "gate_lora_b", "down_lora_a", "down_lora_b"]

    for name, auto_grad, manual_grad in zip(gradient_names, auto_grads, manual_grads):
        try:
            torch.testing.assert_close(auto_grad, manual_grad)
            print(f"Gradient match for {name}: PASSED")
        except AssertionError as e:
            print(f"Gradient match for {name}: FAILED")
            print(e)

if __name__ == '__main__':

    test_transformer_block_backward_manual_class()
    test_lora_transformer_block_backward_manual_class()
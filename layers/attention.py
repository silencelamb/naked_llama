import torch
from torch import nn
import math
# from .rope import get_rope_embeddings, apply_rotary_pos_emb, apply_rotary_pos_emb_backward
from layers.rope import get_rope_embeddings, apply_rotary_pos_emb, apply_rotary_pos_emb_backward, init_rope_embeddings
from layers.matmul import MLP
from layers.activation import Softmax
from configuration_llama import LlamaConfig
from utils import get_attentioin_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_kv_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)



class ScaledDotProductAttention():
    def __init__(self, ):
        self.cache = None
        self.softmax = Softmax()
    
    def forward(self, query, key, value, attention_mask):
        head_dim = query.size(-1)  # 获取 头的维度
        # transpose key
        transposed_key = key.transpose(2, 3)
        # dot product
        scores = torch.matmul(query, transposed_key)
        # scaled
        scaled_scores = scores / math.sqrt(head_dim)
        
        # one line code : attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(head_dim)
        
        if attention_mask is not None:
            scaled_scores = scaled_scores + attention_mask
        # softmax
        attention_weights = self.softmax.forward(scaled_scores, dtype=torch.float32)
        # weighted sum
        output = torch.matmul(attention_weights, value)
        self.cache = (query, key, value, scaled_scores, attention_weights)
        return output

    def backward(self, grad_output):
        query, key, value, scaled_scores, attention_weights = self.cache
        head_dim = query.size(-1)  
    
        # 反向传播
        grad_atten_w = torch.matmul(grad_output, value.transpose(-2, -1))
        grad_scaled_scores = self.softmax.backward(grad_atten_w)
        # Mask 不影响梯度
        grad_scores = grad_scaled_scores / math.sqrt(head_dim)
    
        if query.requires_grad:
            # query梯度
            grad_query = torch.matmul(grad_scores, key)
            
        if key.requires_grad:
            # key梯度
            grad_key = torch.matmul(grad_scores.transpose(-2, -1), query)

        if value.requires_grad:
            # value梯度
            grad_value = torch.matmul(attention_weights.transpose(-2, -1), grad_output)
    
        return grad_query, grad_key, grad_value

class LlamaAttention():
    def __init__(self, config: LlamaConfig, w_q, w_k, w_v, w_o, bias_q=None, bias_k=None, bias_v=None, bias_o=None):
        self.q_proj = MLP(w_q, bias_q)
        self.k_proj = MLP(w_k, bias_k)
        self.v_proj = MLP(w_v, bias_v)
        self.o_proj = MLP(w_o, bias_o)
        self.scale_dot_product_attention = ScaledDotProductAttention()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.cache = None
    
    def forward(self, hidden_states, mask):
        batch_size, seq_len, hidden_size = hidden_states.shape[0:3]
        # 线性变换
        query = self.q_proj.forward(hidden_states)
        key = self.k_proj.forward(hidden_states)
        value = self.v_proj.forward(hidden_states)
    
        head_dim = query.shape[2] // self.num_heads
        # 将Q, K, V矩阵分割成多个头, [batch_size, heads, seq_len, head_dim]
        # 先 view， 然后transpose
        query = query.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, head_dim).transpose(1, 2)
        
        # ROPE计算
        cos, sin = get_rope_embeddings(value, seq_len=seq_len)
        past_key_values_length = 0
        device = hidden_states.device
        position_ids = torch.arange(
            past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)
        query_rope, key_rope = apply_rotary_pos_emb(query, key, cos, sin, position_ids=position_ids)
        
        # 重复多头, 对于 7B num_kv_groups=32/32=1， 对于 70B num_kv_groups=64/8=8
        key_repeat = repeat_kv(key_rope, self.num_kv_groups)
        value_repeat = repeat_kv(value, self.num_kv_groups)

        # 注意力机制
        attention_output = self.scale_dot_product_attention.forward(query_rope, key_repeat, value_repeat, mask)

        # 重新组合多头的输出
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(batch_size, seq_len, hidden_size)
        
        # 重新组合多头输出，另一种写法
        # attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        # attention_output = attention_output.view(batch_size, seq_len, -1)
        
        # 输出线性变换
        output = self.o_proj.forward(attention_output)   
        self.cache = (hidden_states, query, key, value, cos, sin, position_ids) 
        return output
    
    def backward(self, grad_output):
        # 反向传播
        hidden_states, query, key, value, cos, sin, position_ids = self.cache
        grad_w_q, grad_bias_q = None, None
        grad_w_k, grad_bias_k = None, None
        grad_w_v, grad_bias_k = None, None
        grad_w_o, grad_bias_o = None, None
        batch_size, seq_len, hidden_size = hidden_states.shape
        head_dim = query.shape[-1]
        
        grad_atten_o, grad_w_o, grad_bias_o = self.o_proj.backward(grad_output)
        grad_atten_o = grad_atten_o.view(batch_size, seq_len, self.num_heads, head_dim)
        grad_atten_o = grad_atten_o.transpose(1, 2).contiguous()
        
        # 注意力计算 反向梯度
        grad_query, grad_key, grad_value = self.scale_dot_product_attention.backward(grad_atten_o)
        
        # 重复多头 反向梯度
        grad_key = repeat_kv_backward(grad_key, self.num_kv_groups)
        grad_value = repeat_kv_backward(grad_value, self.num_kv_groups)
        
        # rope 反向梯度
        grad_query, grad_key = apply_rotary_pos_emb_backward(grad_query, grad_key, query, key, cos, sin, position_ids)
        
        grad_query = grad_query.transpose(1, 2).reshape(batch_size * seq_len, head_dim * self.num_heads)
        grad_key = grad_key.transpose(1, 2).reshape(batch_size * seq_len, head_dim * self.num_kv_heads)
        grad_value = grad_value.transpose(1, 2).reshape(batch_size * seq_len, head_dim * self.num_kv_heads)

        #q_proj/k_proj/v_proj 线性变换的梯度
        grad_x_q, grad_w_q, grad_bias_q = self.q_proj.backward(grad_query)
        grad_x_k, grad_w_k, grad_bias_k = self.k_proj.backward(grad_key)
        grad_x_v, grad_w_v, grad_bias_v = self.v_proj.backward(grad_value)
        
        grad_x = grad_x_q + grad_x_k + grad_x_v
        grad_x = grad_x.view(batch_size, seq_len, hidden_size)

        return grad_x, grad_w_q, grad_w_k, grad_w_v, grad_w_o, grad_bias_q, grad_bias_k, grad_bias_v, grad_bias_o
    
### multi_head_attention's backward implementation ###
def repeat_kv_backward(grad_h: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Backward pass for the repeat_kv function.
    """
    batch, num_kv_heads, slen, head_dim = grad_h.shape
    num_kv_heads = num_kv_heads // n_rep
    
    if n_rep == 1:
        return grad_h

    grad_h = grad_h.view(batch, num_kv_heads, n_rep, slen, head_dim)
    grad_h = grad_h.sum(dim=2)
    
    return grad_h

def test_multi_head_attention_backward_manual_class():

    batch_size, seq_len, num_heads, head_dim, num_kv_heads = 1, 6, 8, 8, 8
    x = torch.randn(batch_size, seq_len, num_heads*head_dim)

    # bias is only for test, in real case, llama2 does not use bias, qwen2 uses bias.
    w_q = nn.Parameter(torch.randn(num_heads*head_dim, num_heads*head_dim))
    w_k = nn.Parameter(torch.randn(num_kv_heads*head_dim, num_heads*head_dim))
    w_v = nn.Parameter(torch.randn(num_kv_heads*head_dim, num_heads*head_dim))
    w_o = nn.Parameter(torch.randn(num_heads*head_dim, num_heads*head_dim))
    
    bias_q = nn.Parameter(torch.randn(num_heads*head_dim))
    bias_k = nn.Parameter(torch.randn(num_kv_heads*head_dim))
    bias_v = nn.Parameter(torch.randn(num_kv_heads*head_dim))
    bias_o = nn.Parameter(torch.randn(num_heads*head_dim))

    mask = get_attentioin_mask(start_pos=0, seq_length=seq_len, ref_tensor=x)
    config = LlamaConfig(num_attention_heads=8, num_key_value_heads=8)
    cos_cached, sin_cached = init_rope_embeddings(dim=head_dim, max_position_embeddings=seq_len)

    x.requires_grad_(True)
    w_q.requires_grad_(True)
    w_k.requires_grad_(True)
    w_v.requires_grad_(True)
    w_o.requires_grad_(True)
    bias_q.requires_grad_(True)
    bias_k.requires_grad_(True)
    bias_v.requires_grad_(True)
    bias_o.requires_grad_(True)
    
    multi_head_attention = LlamaAttention(config, w_q, w_k, w_v, w_o, bias_q, bias_k, bias_v, bias_o)

    output = multi_head_attention.forward(x, mask)
    # 定义输出的grad
    grad_output = .1 * torch.randn_like(output)
    output.backward(grad_output, retain_graph=True) 
    
    d_x_class, d_wq_class, d_wk_class, d_wv_class, d_wo_class, d_bias_q_class, d_bias_k_class, d_bias_v_class, \
        d_bias_o_class  = [_.grad.clone() for _ in [x, w_q, w_k, w_v, w_o, bias_q, bias_k, bias_v, bias_o]]
    

    d_x_manual, d_wq_manual, d_wk_manual, d_wv_manual, d_wo_manual, d_bias_q_manual, d_bias_k_manual, \
        d_bias_v_manual, d_bias_o_manual = multi_head_attention.backward(grad_output)
    
    print(torch.testing.assert_close(d_x_class, d_x_manual))
    print(torch.testing.assert_close(d_wq_class, d_wq_manual))
    print(torch.testing.assert_close(d_wk_class, d_wk_manual))
    print(torch.testing.assert_close(d_wv_class, d_wv_manual))
    print(torch.testing.assert_close(d_wo_class, d_wo_manual))
    print(torch.testing.assert_close(d_bias_q_class, d_bias_q_manual))
    print(torch.testing.assert_close(d_bias_k_class, d_bias_k_manual))
    print(torch.testing.assert_close(d_bias_v_class, d_bias_v_manual))
    print(torch.testing.assert_close(d_bias_o_class, d_bias_o_manual))

if __name__ == '__main__':
    torch.manual_seed(65536)
    torch.set_printoptions(linewidth=200)         # 这样打印不会存在折叠的问题
    # 设置测试参数
    batch_size, num_kv_heads, sequence_length, head_dimension = 1, 2, 4, 3
    n_rep = 2  # 重复次数

    # 创建一个随机初始化的隐藏状态张量
    hidden_states = torch.randn(batch_size, num_kv_heads, sequence_length, head_dimension)

    # 调用函数
    output = repeat_kv(hidden_states, n_rep)

    # 打印原始张量和输出张量的形状
    print("原始张量形状:", hidden_states.shape)
    print("输出张量形状:", output.shape)    
    
    # 多头注意力反向梯度测试
    test_multi_head_attention_backward_manual_class()
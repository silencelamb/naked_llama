import torch
from torch import nn
import math
# from .rope import get_rope_embeddings, apply_rotary_pos_emb, apply_rotary_pos_emb_backward
from layers.rope import get_rope_embeddings, apply_rotary_pos_emb, apply_rotary_pos_emb_backward, init_rope_embeddings
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



def scaled_dot_product_attention(query, key, value, attention_mask):
    
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
    attention_weights = torch.softmax(scaled_scores, dim=-1)
    # weighted sum
    output = torch.matmul(attention_weights, value)
    return output

def multi_head_attention(hidden_states, w_q, w_k, w_v, w_o, config: LlamaConfig, mask):
    """
    """
        
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    
    
    batch_size, seq_len, hidden_size = hidden_states.shape[0:3]
    
    # 线性变换
    query = torch.matmul(hidden_states, w_q.T)
    key = torch.matmul(hidden_states, w_k.T)
    value = torch.matmul(hidden_states, w_v.T)
    
    head_dim = query.shape[2] // num_heads
    
    # 将Q, K, V矩阵分割成多个头, [batch_size, heads, seq_len, head_dim]
    # 先 view， 然后transpose
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # 将Q, K, V矩阵分割成多个头,  另一种写法
    # query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    # key = key.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    # value = value.view(batch_size, seq_len, num_kv_heads, head_dim).permute(0, 2, 1, 3)
    
    # ROPE计算
    cos, sin = get_rope_embeddings(value, seq_len=seq_len)
    past_key_values_length = 0

    device = hidden_states.device
    position_ids = torch.arange(
        past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids=position_ids)

    # 重复多头, 对于 7B num_kv_groups=32/32=1， 对于 70B num_kv_groups=64/8=8
    key = repeat_kv(key, num_kv_groups)
    value = repeat_kv(value, num_kv_groups)

    # 注意力机制
    attention_output = scaled_dot_product_attention(query, key, value, mask)

    # 重新组合多头的输出
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(batch_size, seq_len, hidden_size)
    
    # 重新组合多头输出，另一种写法
    # attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
    # attention_output = attention_output.view(batch_size, seq_len, -1)
    
    # 输出线性变换
    output = torch.matmul(attention_output, w_o.T)
    
    return output

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

def softmax_backward(grad_output, x):
    # s = torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True) 
    # 使用稳定实现的softmax
    s = torch.softmax(x, dim=-1)
    grad_x = s * (grad_output - torch.sum(grad_output * s, dim=-1, keepdim=True))
    return grad_x

def scaled_dot_product_attention_backward(grad_output, query, key, value, attention_mask):
    head_dim = query.size(-1)  
    transposed_key = key.transpose(-2, -1)
    
    # 前向传播中的一些中间结果
    scores = torch.matmul(query, transposed_key)
    scaled_scores = scores / math.sqrt(head_dim)
    if attention_mask is not None:
        scaled_scores += attention_mask
    atten_w = torch.softmax(scaled_scores, dim=-1)
    
    # 反向传播
    grad_atten_w = torch.matmul(grad_output, value.transpose(-2, -1))
    grad_scaled_scores = softmax_backward(grad_atten_w, scaled_scores)
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
        grad_value = torch.matmul(atten_w.transpose(-2, -1), grad_output)
    
    return grad_query, grad_key, grad_value
    
    
def multi_head_attention_backward(grad_output, x, w_q, w_k, w_v, w_o, config, mask):
    """
    multi_head_attention 手写反向实现
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_kv_groups = num_heads // num_kv_heads
    
    batch_size, seq_len, hidden_size = x.shape[0:3]
    # 需要部分前向传播的中间结果
    # 线性变换
    query = torch.matmul(x, w_q.T)
    key = torch.matmul(x, w_k.T)
    value = torch.matmul(x, w_v.T)
    
    head_dim = query.shape[2] // num_heads
    
    # 将Q, K, V矩阵分割成多个头, [batch_size, heads, seq_len, head_dim]
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # ROPE计算
    cos, sin = get_rope_embeddings(value, seq_len=seq_len)
    past_key_values_length = 0

    device = x.device
    position_ids = torch.arange(
        past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0)
    e_query, e_key = apply_rotary_pos_emb(query, key, cos, sin, position_ids=position_ids)

    # 重复多头
    r_key = repeat_kv(e_key, num_kv_groups)
    r_value = repeat_kv(value, num_kv_groups)

    # 注意力机制
    atten_o = scaled_dot_product_attention(e_query, r_key, r_value, mask)

    # 重新组合多头的输出
    atten_o = atten_o.transpose(1, 2).contiguous()
    atten_o = atten_o.reshape(batch_size, seq_len, hidden_size)
    
    # 输出线性变换
    # output = torch.matmul(attention_output, w_o.T)
    
    # 反向传播
    grad_w_o = torch.matmul(grad_output.reshape(batch_size*seq_len, hidden_size).T, atten_o.reshape(batch_size*seq_len, hidden_size))
    grad_atten_o = torch.matmul(grad_output, w_o)
    grad_atten_o = grad_atten_o.view(batch_size, seq_len, num_heads, head_dim)
    grad_atten_o = grad_atten_o.transpose(1, 2).contiguous()
    
    # 注意力计算 反向梯度
    grad_query, grad_key, grad_value = scaled_dot_product_attention_backward(grad_atten_o, e_query, r_key, r_value, mask)
    
    # 重复多头 反向梯度
    grad_key = repeat_kv_backward(grad_key, num_kv_groups)
    grad_value = repeat_kv_backward(grad_value, num_kv_groups)
    
    # rope 反向梯度
    grad_query, grad_key = apply_rotary_pos_emb_backward(grad_query, grad_key, query, key, cos, sin, position_ids)
    
    grad_query = grad_query.transpose(1, 2).reshape(batch_size * seq_len, head_dim * num_heads)
    grad_key = grad_key.transpose(1, 2).reshape(batch_size * seq_len, head_dim * num_kv_heads)
    grad_value = grad_value.transpose(1, 2).reshape(batch_size * seq_len, head_dim * num_kv_heads)

    # 线性变换的梯度
    if w_q.requires_grad:
        grad_w_q = torch.matmul(grad_query.T, x.reshape(batch_size * seq_len, hidden_size))
    if w_k.requires_grad:
        grad_w_k = torch.matmul(grad_key.T, x.reshape(batch_size * seq_len, hidden_size))
    if w_v.requires_grad:    
        grad_w_v = torch.matmul(grad_value.T, x.reshape(batch_size * seq_len, hidden_size))
    
    grad_x_q = torch.matmul(grad_query, w_q)
    grad_x_k = torch.matmul(grad_key, w_k)
    grad_x_v = torch.matmul(grad_value, w_v)
    
    grad_x = grad_x_q + grad_x_k + grad_x_v
    grad_x = grad_x.view(batch_size, seq_len, hidden_size)

    return grad_x, grad_w_q, grad_w_k, grad_w_v, grad_w_o

def test_multi_head_attention_backward_manual_func():

    batch_size, seq_len, num_heads, head_dim, num_kv_heads = 1, 6, 8, 8, 8
    x = torch.randn(batch_size, seq_len, num_heads*head_dim)

    w_q = nn.Parameter(torch.randn(num_heads*head_dim, num_heads*head_dim))
    w_k = nn.Parameter(torch.randn(num_kv_heads*head_dim, num_heads*head_dim))
    w_v = nn.Parameter(torch.randn(num_kv_heads*head_dim, num_heads*head_dim))
    w_o = nn.Parameter(torch.randn(num_heads*head_dim, num_heads*head_dim))

    mask = get_attentioin_mask(start_pos=0, seq_length=seq_len, ref_tensor=x)
    config = LlamaConfig(num_attention_heads=8, num_key_value_heads=8)
    cos_cached, sin_cached = init_rope_embeddings(dim=head_dim, max_position_embeddings=seq_len)

    x.requires_grad_(True)
    w_q.requires_grad_(True)
    w_k.requires_grad_(True)
    w_v.requires_grad_(True)
    w_o.requires_grad_(True)

    output = multi_head_attention(x, w_q, w_k, w_v, w_o, config, mask)
    # 定义输出的grad
    grad_output = .1 * torch.randn_like(output)
    output.backward(grad_output, retain_graph=True) 
    
    d_x_class, d_wq_class, d_wk_class, d_wv_class,d_wo_class = [_.grad.clone() for _ in [x, w_q, w_k, w_v, w_o]]
    
    x = x.clone().detach().requires_grad_(True)
    w_q = w_q.clone().detach().requires_grad_(True)
    w_k = w_k.clone().detach().requires_grad_(True)
    w_v = w_v.clone().detach().requires_grad_(True)
    w_o = w_o.clone().detach().requires_grad_(True)

    d_x_manual, d_wq_manual, d_wk_manual, d_wv_manual,d_wo_manual = \
        multi_head_attention_backward(grad_output, x, w_q, w_k, w_v, w_o, config, mask)
    
    print(torch.testing.assert_close(d_wo_class, d_wo_manual))
    print(torch.testing.assert_close(d_wq_class, d_wq_manual))
    print(torch.testing.assert_close(d_x_class, d_x_manual))
    print(torch.testing.assert_close(d_wk_class, d_wk_manual))
    print(torch.testing.assert_close(d_wv_class, d_wv_manual))

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
    test_multi_head_attention_backward_manual_func()
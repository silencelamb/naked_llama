import torch
import math
from .rope import get_rope_embeddings, apply_rotary_pos_emb

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

def multi_head_attention(hidden_states, w_q, w_k, w_v, w_o, num_heads, mask):
    """
    """
    batch_size, seq_len, hidden_size = hidden_states.shape[0:3]
    
    # 线性变换
    query = torch.matmul(hidden_states, w_q)
    key = torch.matmul(hidden_states, w_k)
    value = torch.matmul(hidden_states, w_v)
    
    head_dim = query.shape[2] // num_heads
    
    
    # 将Q, K, V矩阵分割成多个头, [batch_size, heads, seq_len, head_dim]
    # 先 view， 然后transpose
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # 将Q, K, V矩阵分割成多个头,  另一种写法
    # query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    # key = key.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    # value = value.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    
    # ROPE计算
    cos, sin = get_rope_embeddings(value, seq_len=seq_len)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids=None)

    # 注意力机制
    attention_output = scaled_dot_product_attention(query, key, value, mask)
    
    # 重新组合多头的输出
    attention_output = attention_output.transpose(1, 2).contiguous()
    attention_output = attention_output.reshape(batch_size, seq_len, hidden_size)
    
    # 重新组合多头输出，另一种写法
    # attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
    # attention_output = attention_output.view(batch_size, seq_len, -1)
    
    # 输出线性变换
    output = torch.matmul(attention_output, w_o)
    
    return output

if __name__ == '__main__':
    pass
    # test case
import torch
from rope import get_rotary_emb, apply_rotary_pos_emb

def scaled_dot_product_attention(query, key, value):
    d_k = query.size(-1)  # 获取 头 的深度
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output



def multi_head_attention(hidden_states, w_q, w_k, w_v, w_o, num_heads):
    """
    """
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    # 线性变换
    query = torch.matmul(hidden_states, w_q)
    key = torch.matmul(hidden_states, w_k)
    value = torch.matmul(value, w_v)
    

    depth = query.shape[2] // num_heads

    # 将权重矩阵分割成多个头并应用于 Q, K, V
    query = query.view(batch_size, seq_len, num_heads, depth).permute(0, 2, 1, 3)
    key = key.view(batch_size, seq_len, num_heads, depth).permute(0, 2, 1, 3)
    value = value.view(batch_size, seq_len, num_heads, depth).permute(0, 2, 1, 3)


    cos, sin = get_rotary_emb(value, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    
    # 注意力机制
    attention_output = scaled_dot_product_attention(query, key, value)
    
    # 重新组合头
    attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
    attention_output = attention_output.view(batch_size, seq_len, -1)
    
    # 输出线性变换
    output = torch.matmul(attention_output, w_o)
    
    return output

if __name__ == '__main__':
    # test case
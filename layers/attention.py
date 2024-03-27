import torch
import math
from .rope import get_rope_embeddings, apply_rotary_pos_emb
from configuration_llama import LlamaConfig


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



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
    num_key_value_heads = config.num_key_value_heads
    num_key_value_groups = num_heads // num_key_value_heads
    
    
    batch_size, seq_len, hidden_size = hidden_states.shape[0:3]
    
    # 线性变换
    query = torch.matmul(hidden_states, w_q.T)
    key = torch.matmul(hidden_states, w_k.T)
    value = torch.matmul(hidden_states, w_v.T)
    
    head_dim = query.shape[2] // num_heads
    
    # 将Q, K, V矩阵分割成多个头, [batch_size, heads, seq_len, head_dim]
    # 先 view， 然后transpose
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2)
    
    # 将Q, K, V矩阵分割成多个头,  另一种写法
    # query = query.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    # key = key.view(batch_size, seq_len, num_key_value_heads, head_dim).permute(0, 2, 1, 3)
    # value = value.view(batch_size, seq_len, num_key_value_heads, head_dim).permute(0, 2, 1, 3)
    
    # ROPE计算
    cos, sin = get_rope_embeddings(value, seq_len=seq_len)
    past_key_values_length = 0

    device = hidden_states.device
    position_ids = torch.arange(
        past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
    )
    position_ids = position_ids.unsqueeze(0)
    query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids=position_ids)

    # 重复多头, 对于 7B num_key_value_groups=32/32=1， 对于 70B num_key_value_groups=64/8=8
    key = repeat_kv(key, num_key_value_groups)
    value = repeat_kv(value, num_key_value_groups)

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

if __name__ == '__main__':
    pass
    # test case
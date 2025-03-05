# We calculate the number of float ops needed by the part of MLA computation graph,
# input tensors: c_Q and cached k_pe and compressed_kv
# output tensor: the output hidden states

# We omitted the calculation from input hidden states to c_Q and cached k_pe and compressed_kv, 
# because it's the same for both vanilla version and mat-absorb version.

hidden_dim = 7168
head_num = 128
head_dim_v = 128
rope_dim = 64
head_dim_qk = head_dim_v + rope_dim
lora_rank_k = 512
lora_rank_q = 1536


kv_len = 100


# vanilla version
def num_float_ops_vanilla(q_len, kv_len):
    return ( q_len * lora_rank_q * head_num * (head_dim_v + rope_dim) + # from c_Q to q_pe and q_nope, corresponding to q_b_proj
                kv_len * head_num * lora_rank_k * (head_dim_v + head_dim_v) +           # from compressed_kv to k_nop and value_states, corresponding to kv_b_proj
                head_num * (q_len * head_dim_qk * kv_len +  q_len * kv_len * head_dim_v) +  # 128 heads MHA
                q_len * (head_num*head_dim_v) * hidden_dim) # from MHA output to output hidden states, corresponding to o_proj

def mem_footprint_vanilla(q_len, kv_len):
    return ( q_len*lora_rank_q + lora_rank_q*(head_num*head_dim_qk) +    # q_lora, q_up_weight
                kv_len*(lora_rank_k+rope_dim) + lora_rank_k*(head_num*(head_dim_v+head_dim_v)) +  # cached_k_lora, W_UK,  W_UV  
                head_num * (q_len * head_dim_qk + head_dim_qk * kv_len + head_dim_v * kv_len) +  # 128 heads MHA,  q, k, v
                q_len * (head_num*head_dim_v) + (head_num*head_dim_v) * hidden_dim )   # attn_output,  o_proj weight

# absorbed version weight multiply
def num_float_ops_mat_absorb_mul(q_len, kv_len):
    return ( q_len * lora_rank_q * head_num * (head_dim_v + rope_dim) + # from c_Q to q_pe and q_nope, corresponding to q_b_proj
                q_len * head_num * head_dim_v * lora_rank_k + # from q_nope to q_nope 512 dim, corresponding to W_UK
                head_num * (q_len * (lora_rank_k+rope_dim) * kv_len +  q_len * kv_len * lora_rank_k) + # 128 heads MQA
                q_len * head_num * lora_rank_k * head_dim_v  +  # MHA output 512 dim => 128 dim, corresponding to W_UV_O
                q_len * head_num * head_dim_v * hidden_dim) #  # from MHA output to output hidden states, corresponding to o_proj

def mem_footprint_mat_absorb_mul(q_len, kv_len):
    return ( q_len*lora_rank_q + lora_rank_q*(head_num*head_dim_qk) +    # q_lora, q_up_weight
                q_len * (head_num*head_dim_qk) + # q dim 192
                kv_len*(lora_rank_k+rope_dim) + lora_rank_k*(head_num*(head_dim_v+head_dim_v)) +  # cached_k_lora, W_UK,  W_UV 
                head_num * (q_len* (lora_rank_k+rope_dim)) +  # 128 heads Q
                q_len * head_num * lora_rank_k + # atten output 512 dim
                q_len * (head_num*head_dim_v) + (head_num*head_dim_v) * hidden_dim )   # attn_output,  o_proj weight


# absorbed version full absorb
def num_float_ops_mat_absorb_all(q_len, kv_len):
    return ( q_len * lora_rank_q * head_num * (head_dim_v + rope_dim) + # from c_Q to q_pe and q_nope, corresponding to q_b_proj
                q_len* head_num * lora_rank_q * lora_rank_k + # from c_Q to q_nope, corresponding to W_UQUK
                head_num * (q_len * (lora_rank_k+rope_dim) * kv_len +  q_len * kv_len * lora_rank_k) + # 128 heads MQA
                q_len * head_num * lora_rank_k * hidden_dim) # from MHA output to output hidden states, corresponding to W_UV_O


def mem_footprint_mat_absorb_all(q_len, kv_len):
    return ( q_len*lora_rank_q + lora_rank_q*head_num*rope_dim +    # q_lora, q_rope_weight
                q_len * (head_num*rope_dim) + # qrope
                head_num*hidden_dim*lora_rank_k + # W_UQUK
                kv_len*(lora_rank_k+rope_dim)  +  # cached_k_lora
                head_num * q_len * (lora_rank_k+rope_dim) + # 128 heads Q 
                q_len * (head_num*lora_rank_k) + # attn output
                (head_num*lora_rank_k) * hidden_dim) # W_UV_O


print(f"q_len: {q_len} , kv_len: {kv_len} ")
print('\n' + '='*90 + '\n')

print(f"prefill: num_float_ops mat_absorb (multiply) vs vanilla ratio  ~ {num_float_ops_mat_absorb_mul(kv_len, kv_len) / num_float_ops_vanilla(kv_len, kv_len):.5f}")
print(f"prefill: mem_footprint mat_absorb (multiply) vs vanilla ratio  ~ {mem_footprint_mat_absorb_mul(kv_len, kv_len) / mem_footprint_vanilla(kv_len, kv_len):.5f}\n")
print(f"decode: num_float_ops mat_absorb (multiply) vs vanilla ratio  ~ {num_float_ops_mat_absorb_mul(1, kv_len) / num_float_ops_vanilla(1, kv_len):.5f}")
print(f"decode: mem_footprint mat_absorb (multiply) vs vanilla ratio  ~ {mem_footprint_mat_absorb_mul(1, kv_len) / mem_footprint_vanilla(1, kv_len):.5f}")

print('\n' + '='*90 + '\n')

print(f"prefill: num_float_ops mat_absorb (full absorbed) vs vanilla ratio  ~ {num_float_ops_mat_absorb_all(kv_len, kv_len) / num_float_ops_vanilla(kv_len, kv_len):.5f}")
print(f"prefill: mem_footprint mat_absorb (full absorbed) vs vanilla ratio  ~ {mem_footprint_mat_absorb_all(kv_len, kv_len) / mem_footprint_vanilla(kv_len, kv_len):.5f}\n")
print(f"decode: num_float_ops mat_absorb (full absorbed) vs vanilla ratio  ~ {num_float_ops_mat_absorb_all(1, kv_len) / num_float_ops_vanilla(1, kv_len):.5f}")
print(f"decode: mem_footprint mat_absorb (full absorbed) vs vanilla ratio  ~ {mem_footprint_mat_absorb_all(1, kv_len) / mem_footprint_vanilla(1, kv_len):.5f}")

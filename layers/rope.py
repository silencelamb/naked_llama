import torch

cos_cached = None
sin_cached = None


def init_rope_embeddings(dim, max_position_embeddings=4096, base=10000, device=None, scaling_factor=1.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    t = torch.arange(max_position_embeddings, device=device, dtype=torch.int64).type_as(inv_freq)
    t = t / scaling_factor
    freqs = torch.outer(t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    global cos_cached, sin_cached
    cos_cached = emb.cos().to(torch.get_default_dtype())
    sin_cached = emb.sin().to(torch.get_default_dtype())
    return cos_cached,sin_cached


def get_rope_embeddings(x, seq_len=None):
    # x: [bs, num_attention_heads, seq_len, head_size]
    global cos_cached, sin_cached
    return (
        cos_cached[:seq_len].to(dtype=x.dtype),
        sin_cached[:seq_len].to(dtype=x.dtype),
    )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    # properly broadcast cos and sin to the dimensions of q and k
    # cos shape torch.Size([seq_len = 13, head_dim = 128])
    # cos[position_ids] shape torch.Size([1, seq_len = 13, head_dim = 128])
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    # cos shape torch.Size([1, 1, seq_len = 13, head_dim = 128])
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # q shape torch.Size([batch size = 1, num_head = 32, seq_len = 13, head_dim = 128])
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

if __name__ == '__main__':
    init_rope_embeddings(dim=4096)
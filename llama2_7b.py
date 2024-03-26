import torch
from utils import npy_to_tensor
from layers.norm import RMSNorm
from layers.embedding import embedding_lookup
from layers.matmul import LlamaMLP, lm_head
from layers.transformer_block import llama2_transformer_block


def llama2_7b(token_ids: torch.Tensor):
    """
    手动实现llama2 7b的推理计算。
    
    参数:
    - token_ids: token id组成的tensor，形状为 [batch_size, seq_length]
    """
    bsz, seq_length = token_ids.shape()
    
    # em  
    embding_weights = npy_to_tensor('weights/llama2_7b/model.embed_tokens.weight.npy')
    input_embeds = embedding_lookup(token_ids, embding_weights)
    hidden_states = input_embeds  # shape [batch_size, seq_length, hidden_size], hidden_size=4096
    # if seq_length > 1:
    #     mask = torch.full(
    #         (seq_length, seq_length), float("-inf"), device=token_ids.device
    #     )

    #     mask = torch.triu(mask, diagonal=1)

    #     # When performing key-value caching, we compute the attention scores
    #     # only for the new sequence. Thus, the matrix of scores is of size
    #     # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
    #     # j > cache_len + i, since row i corresponds to token cache_len + i.
    #     mask = torch.hstack([
    #         torch.zeros((seqlen, start_pos), device=tokens.device),
    #         mask
    #     ]).type_as(h)

    
    # 重复32次 llama2_transformer_block 的计算
    
    for layer_id in range(32):
        output = llama2_transformer_block(hidden_states, mask=None, num_heads=None, layer_id=layer_id)
        hidden_states = output[0]
    
    hidden_states = RMSNorm(hidden_states, layer_id)
    logits = lm_head(hidden_states)
    return logits
    
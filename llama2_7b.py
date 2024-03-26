import torch
from utils import npy_to_tensor
from layers.norm import RMSNorm
from layers.embedding import embedding_lookup
from layers.matmul import LlamaMLP, lm_head
from layers.transformer_block import llama2_transformer_block
from transformers import AutoTokenizer, LlamaForCausalLM


def llama2_7b(token_ids: torch.Tensor):
    """
    手动实现llama2 7b的推理计算。
    
    参数:
    - token_ids: token id组成的tensor，形状为 [batch_size, seq_length]
    """
    bsz, seq_length = token_ids.shape
    
    
    # embedding 
    embdding_weights = npy_to_tensor('weights/llama2_7b/model.embed_tokens.weight.npy')
    input_embeds = embedding_lookup(token_ids, embdding_weights)
    hidden_states = input_embeds  # shape [batch_size, seq_length, hidden_size], hidden_size=4096
    
    # mask
    start_pos = 0
    if seq_length > 1:
        mask = torch.full(
            (seq_length, seq_length), float("-inf"), device=token_ids.device
        )

        mask = torch.triu(mask, diagonal=1)

        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack([
            torch.zeros((seq_length, start_pos), device=token_ids.device),
            mask
        ]).type_as(hidden_states)

    # 重复32次 llama2_transformer_block 的计算
    for layer_id in range(32):
        output = llama2_transformer_block(hidden_states, num_heads=32, layer_id=layer_id, attention_mask=mask)
        hidden_states = output[0]
    
    hidden_states = RMSNorm(hidden_states, layer_id)
    lm_head_weight = npy_to_tensor('weights/llama2_7b/lm_head.weight.npy')
    logits = lm_head(hidden_states, lm_head_weight)
    return logits


if __name__ == '__main__':
    # test case
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids

    # random input
    # token_ids = torch.randint(0, 32000, (1, 1024))
    # token_ids = torch.repeat_interleave(token_ids, 2, dim=0)
    # token_ids = torch.repeat_interleave(token_ids, 4, dim=1) # (2, 52)

    # check result
    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    # cpu_res = llama(input_ids).logits
    
    logits = llama2_7b(token_ids)
    print(logits.shape)
    print(logits)


    
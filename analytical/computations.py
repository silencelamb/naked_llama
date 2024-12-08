from prettytable import PrettyTable
TFLOPs = 1e12

def analyze_pretrain_computation(batch_size, seq_len, vocab_size, hidden_size, head_num, kv_head, immediate_size, layer_num, mlp_style='llama'):
    '''
    count and analyze computations of LLM pretrain/finetune, pretrain or finetune is the same
    Reference:
    'Efficient large-scale language model training on GPU clusters using megatron-LM' 
    https://dl.acm.org/doi/10.1145/3458817.3476209, page 11, APPENDIX: FLOATING-POINT OPERATIONS
    
    Parameters:
    - batch_size (int): The batch size.
    - seq_len (int): The sequence length.
    - vocab_size (int): The vocabulary size.
    - hidden_size (int): The hidden size.
    - head_num (int): The number of attention heads.
    - kv_head (int): The number of key-value heads.
    - immediate_size (int): The immediate size, ffn up dim.
    - layer_num (int): The number of layers.
    - mlp_style (str): The style of mlp, 'llama' or 'normal'.

    Returns:
    - dict: The computation analysis result.
    '''
    computations = {}
    kv_dim = hidden_size // head_num * kv_head
    # forward pass
    computations['q_proj'] = batch_size * 2 * seq_len * hidden_size * hidden_size
    computations['k_proj'] = batch_size * 2 * seq_len * hidden_size * kv_dim
    computations['v_proj'] = batch_size * 2 * seq_len * hidden_size * kv_dim
    computations['att_q_k'] = batch_size * 2 *  seq_len * seq_len * hidden_size
    computations['att_v'] = batch_size * 2 * seq_len * seq_len * hidden_size
    computations['o_proj'] = batch_size * 2 * seq_len * hidden_size * hidden_size
    computations['mlp_up'] = batch_size * 2 * seq_len * hidden_size * immediate_size
    computations['mlp_down'] = batch_size * 2 * seq_len * immediate_size * hidden_size
    
    if mlp_style == 'llama':        
        computations['mlp_gate'] = batch_size * 2 * seq_len * hidden_size * immediate_size
    else:
        computations['mlp_gate'] = 0
    # multiply layer_num
    for k, v in computations.items():
        computations[k] = v * layer_num

    # add lm_head
    computations['lm_head'] = batch_size * 2 * seq_len * hidden_size * vocab_size
    # backward pass is two times of forward pass
    for k, v in computations.items():
        computations[k] = 3 * v
    computations['total'] = sum(computations.values())
    return computations


def analyze_lora_computation(batch_size, seq_len, vocab_size, hidden_size, head_num, \
    kv_head, immediate_size, layer_num, lora_r, mlp_style='llama'):
    '''
    count and analyze computations of LLM lora finetuning
    Reference:
    'Efficient large-scale language model training on GPU clusters using megatron-LM' 
    https://dl.acm.org/doi/10.1145/3458817.3476209, page 11, APPENDIX: FLOATING-POINT OPERATIONS
    
    Parameters:
    - batch_size (int): The batch size.
    - seq_len (int): The sequence length.
    - vocab_size (int): The vocabulary size.
    - hidden_size (int): The hidden size.
    - head_num (int): The number of attention heads.
    - kv_head (int): The number of key-value heads.
    - immediate_size (int): The immediate size, ffn up dim.
    - layer_num (int): The number of layers.
    - lora_r (int): The lora_r param.
    - mlp_style (str): The style of mlp, 'llama' or 'normal'.

    Returns:
    - dict: The computation analysis result.
    '''
    computations = {}
    kv_dim = hidden_size // head_num * kv_head
    # forward pass
    computations['q_proj'] = batch_size * 2 * seq_len * hidden_size * hidden_size
    computations['k_proj'] = batch_size * 2 * seq_len * hidden_size * kv_dim
    computations['v_proj'] = batch_size * 2 * seq_len * hidden_size * kv_dim
    computations['att_q_k'] = batch_size * 2 *  seq_len * seq_len * hidden_size
    computations['att_v'] = batch_size * 2 * seq_len * seq_len * hidden_size
    computations['o_proj'] = batch_size * 2 * seq_len * hidden_size * hidden_size

    computations['lora_q'] = batch_size * 2 * seq_len * hidden_size * lora_r * 2
    computations['lora_k'] = batch_size * (2 * seq_len * hidden_size * lora_r + 2 * seq_len * lora_r * kv_dim)
    computations['lora_v'] = batch_size * (2 * seq_len * hidden_size * lora_r + 2 * seq_len * lora_r * kv_dim)
    computations['lora_o'] = batch_size * 2 * seq_len * hidden_size * lora_r * 2
    
    computations['mlp_up'] = batch_size * 2 * seq_len * hidden_size * immediate_size
    computations['mlp_down'] = batch_size * 2 * seq_len * immediate_size * hidden_size
    
    computations['lora_mlp_up'] = batch_size * (2 * seq_len * hidden_size * lora_r + 2 * seq_len * lora_r * immediate_size)
    computations['lora_mlp_down'] = batch_size * (2 * seq_len * immediate_size * lora_r + 2 * seq_len * lora_r * hidden_size)
    
    if mlp_style == 'llama':        
        computations['mlp_gate'] = batch_size * 2 * seq_len * hidden_size * immediate_size
        computations['lora_mlp_gate'] = batch_size * (2 * seq_len * hidden_size * lora_r + 2 * seq_len * lora_r * immediate_size)
    else:
        computations['lora_mlp_gate'] = computations['mlp_gate'] = 0
    
    # multiply layer_num
    for k, v in computations.items():
        computations[k] = v * layer_num

    # add lm_head
    computations['lm_head'] = batch_size * 2 * seq_len * hidden_size * vocab_size

    for k, v in computations.items():
        if 'lora' in k:
            # for lora part backward pass is three times of forward pass
            computations[k] = 3 * v
        else:
            # for other part backward pass is two times of forward pass
            computations[k] = 2 * v
    computations['total'] = sum(computations.values())
    return computations


if __name__ == '__main__':

    # llama2 7B, pretrain/SFT
    batch_size, seq_len = 1, 4096
    print(f'===== llama2-7B pretrain/SFT********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    llama2_computations = analyze_pretrain_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=32000, hidden_size=4096, \
        head_num=32, kv_head=32, immediate_size=11008, layer_num=32)
    table = PrettyTable()
    table.field_names = ["name", "TFLOPs"]
    for k, v in llama2_computations.items():
        table.add_row([k, f'{v/TFLOPs: 0.4f}'])
    print(table)
        
    # Qwen2 7B, pretrain/SFT
    batch_size, seq_len = 1, 2048
    print(f'===== Qwen-2  7B pretrain/SFT********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    qwen2_computations = analyze_pretrain_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28)
    table = PrettyTable()
    table.field_names = ["name", "TFLOPs"]
    for k, v in qwen2_computations.items():
        table.add_row([k, f'{v/TFLOPs: 0.4f}'])
    print(table)

    # Qwen2 7B, pretrain/SFT
    batch_size, seq_len = 1, 512
    print(f'===== Qwen-2  7B pretrain/SFT********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    qwen2_computations = analyze_pretrain_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28)
    table = PrettyTable()
    table.field_names = ["name", "TFLOPs"]
    for k, v in qwen2_computations.items():
        table.add_row([k, f'{v/TFLOPs: 0.4f}'])
    print(table)

    # Qwen2 7B, lora finetuning
    batch_size, seq_len = 1, 512
    print(f'===== Qwen-2  7B lora finetuning********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    qwen2_computations = analyze_lora_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28, lora_r=64)
    # table = PrettyTable()
    # table.field_names = ["name", "TFLOPs"]
    # for k, v in qwen2_computations.items():
    #     table.add_row([k, f'{v/TFLOPs: 0.4f}'])
    # print(table)
    print(qwen2_computations['total']/TFLOPs)

    batch_size, seq_len = 1, 1024
    print(f'===== Qwen-2  7B lora finetuning********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    qwen2_computations = analyze_lora_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28, lora_r=64)
    print(qwen2_computations['total']/TFLOPs)

    batch_size, seq_len = 1, 2048
    print(f'===== Qwen-2  7B lora finetuning********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    qwen2_computations = analyze_lora_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28, lora_r=64)
    print(qwen2_computations['total']/TFLOPs)
    
    batch_size, seq_len = 1, 4096
    print(f'===== Qwen-2  7B lora finetuning********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    qwen2_computations = analyze_lora_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28, lora_r=64)
    print(qwen2_computations['total']/TFLOPs)
    
    # llama3 70B, pretrain
    batch_size, seq_len = 1, 8192
    print(f'===== LLaMA3 70B pretrain ********* batch_size: {batch_size}, seq_len: {seq_len} =====')
    llama3_computation = analyze_pretrain_computation(batch_size=batch_size, seq_len=seq_len, vocab_size=128256, hidden_size=8192, \
        head_num=64, kv_head=8, immediate_size=28672, layer_num=80)
    print(f"{llama3_computation['total']/TFLOPs: 0.4f} TFLOPs")
    table = PrettyTable()
    table.field_names = ["name", "TFLOPs"]
    for k, v in llama3_computation.items():
        table.add_row([k, f'{v/TFLOPs: 0.4f}'])
    print(table)





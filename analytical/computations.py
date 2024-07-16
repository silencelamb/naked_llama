from prettytable import PrettyTable
TOPS = 1e12

def analyze_training_computation(batch_size, seq_len, vocab_size, hidden_size, head_num, kv_head, immediate_size, layer_num, mlp_style='llama'):
    '''
    count and analyze computations of LLM model training
    Reference:
    'Efficient large-scale language model training on GPU clusters using megatron-LM' 
    https://dl.acm.org/doi/10.1145/3458817.3476209, page11, APPENDIX: FLOATING-POINT OPERATIONS
    
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
    - int: The total computations (FLOPs).
    '''
    compuations = {}
    # forward pass
    compuations['q_proj'] = batch_size * 2 * seq_len * hidden_size * hidden_size
    compuations['k_proj'] = batch_size * 2 * seq_len * hidden_size * (hidden_size // head_num * kv_head)
    compuations['v_proj'] = batch_size * 2 * seq_len * hidden_size * (hidden_size // head_num * kv_head)
    compuations['att_q_k'] = batch_size * 2 *  seq_len * seq_len * hidden_size
    compuations['att_v'] = batch_size * 2 * seq_len * seq_len * hidden_size
    compuations['o_proj'] = batch_size * 2 * seq_len * hidden_size * hidden_size
    compuations['mlp_up'] = batch_size * 2 * seq_len * hidden_size * immediate_size
    compuations['mlp_down'] = batch_size * 2 * seq_len * immediate_size * hidden_size
    
    if mlp_style == 'llama':        
        compuations['mlp_gate'] = batch_size * 2 * seq_len * hidden_size * immediate_size
    else:
        compuations['mlp_gate'] = 0
    # multiply layer_num
    for k, v in compuations.items():
        compuations[k] = v * layer_num

    # add lm_head
    compuations['lm_head'] = batch_size * 2 * seq_len * hidden_size * vocab_size
    # backward pass is two times of forward pass
    for k, v in compuations.items():
        compuations[k] = 3 * v
    compuations['total'] = sum(compuations.values())
    return compuations


if __name__ == '__main__':

    # llama2 7B
    print('llama2-7B ===============>>>>>>>>>>>>>>>')
    llama2_compuations = analyze_training_computation(batch_size=1, seq_len=4096, vocab_size=32000, hidden_size=4096, \
        head_num=32, kv_head=32, immediate_size=11008, layer_num=32)
    table = PrettyTable()
    table.field_names = ["name", "TOPS"]
    for k, v in llama2_compuations.items():
        table.add_row([k, f'{v/TOPS: 0.4f}'])
    print(table)
        
    # Qwen2 7B
    print('Qwe2-7B ===============>>>>>>>>>>>>>>>')
    qwen2_compuations = analyze_training_computation(batch_size=1, seq_len=4096, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28)
    table = PrettyTable()
    table.field_names = ["name", "TOPS"]
    for k, v in qwen2_compuations.items():
        table.add_row([k, f'{v/TOPS: 0.4f}'])
    print(table)

    # Qwen2 7B
    print('Qwe2-7B ===============>>>>>>>>>>>>>>>')
    qwen2_compuations = analyze_training_computation(batch_size=2, seq_len=512, vocab_size=152064, hidden_size=3584, \
        head_num=28, kv_head=4, immediate_size=18944, layer_num=28)
    table = PrettyTable()
    table.field_names = ["name", "TOPS"]
    for k, v in qwen2_compuations.items():
        table.add_row([k, f'{v/TOPS: 0.4f}'])
    print(table)

from hw_params import GB
DTYPES_BYTES = {'float32': 4,  'float16': 2, 'bfloat16': 2, 'int8': 1}

def llama_activation(head_num, kv_head, batch_size, seq_len, hidden_size, immediate_size, layer_num, vocab_size, is_print=True):
    '''
    Analyze the activations memory of llama2/qwen2 model.
    Reference:
    'Efficient large-scale language model training on GPU clusters using megatron-LM' 
    http://arxiv.org/abs/2205.05198, page 4, 4.1 Activations Memory Per Transformer Layer
    
    Parameters:
    - head_num (int): The number of attention heads.
    - kv_head (int): The number of key-value heads.
    - batch_size (int): The batch size.
    - seq_len (int): The sequence length.
    - hidden_size (int): The hidden size.
    - immediate_size (int): The immediate size, ffn up dim.
    - layer_num (int): The number of layers.
    - vocab_size (int): The vocabulary size.
    - is_print (bool): whether to print the result.

    Returns:
    - dict: The activations memory analysis result.
    '''
    
    qkv_input = DTYPES_BYTES['float16'] * batch_size* seq_len * hidden_size
    q_proj = DTYPES_BYTES['float16'] * batch_size * seq_len * hidden_size
    kv_dim = hidden_size // head_num * kv_head
    k_proj = DTYPES_BYTES['float16'] * batch_size * seq_len * kv_dim
    v_proj = DTYPES_BYTES['float16'] * batch_size * seq_len * kv_dim
    o_proj = DTYPES_BYTES['float16'] * batch_size * seq_len * hidden_size
    softmax_output = DTYPES_BYTES['float32'] * batch_size * head_num * seq_len * seq_len
    norm_input = DTYPES_BYTES['float32'] * batch_size * seq_len * hidden_size * 2  # 2个 RMSNorm
    mlp_up_input = DTYPES_BYTES['float16'] * batch_size * seq_len * hidden_size
    mlp_up_output = DTYPES_BYTES['float16'] * batch_size * seq_len * immediate_size
    mlp_gate_output = DTYPES_BYTES['float16'] * batch_size * seq_len * immediate_size
    mlp_down_input = DTYPES_BYTES['float16'] * batch_size * seq_len * immediate_size
    activation = qkv_input + q_proj + k_proj + v_proj + o_proj + softmax_output + norm_input + mlp_up_input + \
        mlp_up_output + mlp_gate_output + mlp_down_input
    
    last_norm = DTYPES_BYTES['float32'] * batch_size * seq_len * hidden_size
    lm_head_input = DTYPES_BYTES['float16'] * batch_size * seq_len * hidden_size
    last_softmax = DTYPES_BYTES['float32'] * batch_size * seq_len * vocab_size
    total = activation * layer_num + last_norm + lm_head_input + last_softmax
    
    if is_print:
        print(f"head_num={head_num}, kv_head={kv_head}, batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}, layer_num={layer_num}")
        print(f'One Transformer Block:')
        print(f"QKV input: {qkv_input/GB} GB")
        print(f"Q_proj: {q_proj/GB} GB")
        print(f"K_proj: {k_proj/GB} GB")
        print(f"V_proj: {v_proj/GB} GB")
        print(f"O_proj: {o_proj/GB} GB")
        print(f"Softmax output: {softmax_output/GB} GB")
        print(f"RMSNorm input: {norm_input/GB} GB")
        print(f"MLP up input: {mlp_up_input/GB} GB")
        print(f"MLP up output: {mlp_up_output/GB} GB")
        print(f"MLP gate output: {mlp_gate_output/GB} GB")
        print(f"MLP down input: {mlp_down_input/GB} GB")

        print(f"Activation size of One Transformer Block: {activation/GB} GB")
        print(f"Total Activation size: {total/GB} GB")
    return total/GB


if __name__ == '__main__':
    print('llama2 7B activations, seq_len=4096 ==============>')
    llama_activation(head_num=32, kv_head=32, batch_size=1, seq_len=4096, hidden_size=4096, \
        immediate_size=11008, layer_num=32, vocab_size=32000, is_print=False)
    print('llama2 7B activations, seq_len=512 ==============>')
    llama_activation(head_num=32, kv_head=32, batch_size=1, seq_len=512, hidden_size=4096, \
        immediate_size=11008, layer_num=32, vocab_size=32000, is_print=False)
    print('llama2 70B activations==============>')
    llama_activation(head_num=64, kv_head=8, batch_size=1, seq_len=4096, hidden_size=8192, \
        immediate_size=28672, layer_num=80, vocab_size=32000, is_print=False)
    print('Qwen2 7B activations==============>')
    llama_activation(head_num=28, kv_head=4, batch_size=256, seq_len=512, hidden_size=3584, \
        immediate_size=18944, layer_num=28, vocab_size=152064)

    print('Qwen2 7B activations==============>')
    llama_activation(head_num=28, kv_head=4, batch_size=1, seq_len=1024, hidden_size=3584, \
        immediate_size=18944, layer_num=28, vocab_size=152064)
    
    print('Qwen2 7B activations==============>')
    llama_activation(head_num=28, kv_head=4, batch_size=1, seq_len=2048, hidden_size=3584, \
        immediate_size=18944, layer_num=28, vocab_size=152064)
    print('Qwen2 7B activations==============>')
    llama_activation(head_num=28, kv_head=4, batch_size=1, seq_len=4096, hidden_size=3584, \
        immediate_size=18944, layer_num=28, vocab_size=152064)

    print('Qwen2 7B activations==============>')
    llama_activation(head_num=28, kv_head=4, batch_size=256, seq_len=512, hidden_size=3584, \
        immediate_size=18944, layer_num=28, vocab_size=152064)
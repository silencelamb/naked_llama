from transformers import LlamaConfig, AutoModelForCausalLM, LlamaForCausalLM
import torch
from einops import rearrange

intern_llama_mapping = {
    'model.tok_embeddings.weight': 'model.embed_tokens.weight',
    'model.layers.0.attention_norm.weight' : 'model.layers.0.input_layernorm.weight',
    'model.layers.0.attention.wo.weight': 'model.layers.0.self_attn.o_proj.weight',
    'model.layers.0.attention.wqkv.weight': 'model.layers.0.self_attn.qkv_proj.weight',
    'model.layers.0.feed_forward.w1.weight': 'model.layers.0.mlp.gate_proj.weight',
    'model.layers.0.feed_forward.w2.weight': 'model.layers.0.mlp.down_proj.weight',
    'model.layers.0.feed_forward.w3.weight': 'model.layers.0.mlp.up_proj.weight',
    'model.layers.0.ffn_norm.weight': 'model.layers.0.post_attention_layernorm.weight',
    'model.norm.weight': 'model.norm.weight',
    'output.weight': 'lm_head.weight'
}
@torch.no_grad()
def preproces_internlm_qkvweight(qkv_weight, num_key_value_groups, head_dim):
    qkv_weight = qkv_weight.transpose(0, 1).contiguous()
    hidden_dim = qkv_weight.shape[0]
    qkv_weight = rearrange(
        qkv_weight,
        "hidden_dim (h_num gs head_dim) -> hidden_dim h_num gs head_dim",
        gs=2 + num_key_value_groups,
        head_dim=head_dim,
    )
    q_weight = qkv_weight[..., : num_key_value_groups, :]
    q_weight = rearrange(q_weight, "hidden_dim h_num gs head_dim -> hidden_dim (h_num gs) head_dim")
    q_weight = q_weight.reshape((hidden_dim, -1)).transpose(0, 1)

    k_weight = qkv_weight[..., -2, :]
    k_weight = k_weight.reshape((hidden_dim, -1)).transpose(0, 1)

    v_weight = qkv_weight[..., -1, :]
    v_weight = v_weight.reshape((hidden_dim, -1)).transpose(0, 1)
    
    return q_weight, k_weight, v_weight

# 加载 InternLM2.5-7B-Chat 模型
print("Loading internLM weights...")
with torch.no_grad():
    internlm_model = AutoModelForCausalLM.from_pretrained(
        "internlm/internlm2_5-7b-chat",
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
print("Loaded internLM weights!")

# 创建 LLaMA2 格式的配置
llama_config = LlamaConfig(
    vocab_size=internlm_model.config.vocab_size,
    hidden_size=internlm_model.config.hidden_size,
    intermediate_size=internlm_model.config.intermediate_size,
    num_hidden_layers=internlm_model.config.num_hidden_layers,
    num_attention_heads=internlm_model.config.num_attention_heads,
    num_key_value_heads=internlm_model.config.num_key_value_heads,
    max_position_embeddings=internlm_model.config.max_position_embeddings,
    rope_theta=internlm_model.config.rope_theta,
    rms_norm_eps=internlm_model.config.rms_norm_eps,
    torch_dtype=internlm_model.config.torch_dtype,
    bos_token_id=internlm_model.config.bos_token_id,
    eos_token_id=internlm_model.config.eos_token_id,
    pad_token_id=internlm_model.config.pad_token_id,
    rope_scaling=internlm_model.config.rope_scaling,
)

# 转换模型权重
# 提示：internLM中的rope_scaling用的是dynamic, 用的是老版本transformers，key值是 "type"，新版本是 "rope_type"
# 不过没关系，transformer做了向后兼容，type会当做"rope_type"使用，https://github.com/huggingface/transformers/pull/32182
# 转换过程中会提示 Unrecognized keys in `rope_scaling` for 'rope_type'='dynamic': {'type'}，这个没关系
print("Initializing LlaMA using config from InternLM...")
with torch.no_grad():
    llama_model = LlamaForCausalLM(llama_config).to(dtype=torch.bfloat16).to(device=internlm_model.device)

# 复制生成配置，这个挺重要的，因为生成配置会影响到模型的输出对齐
# 关键配置比如是否sample，是否repetition_penalty等
llama_model.generation_config = internlm_model.generation_config

# 复制权重(这里需要根据具体层的对应关系进行修改)
print("Start converting internLM weights to LlaMA...")
head_dim = internlm_model.config.hidden_size // internlm_model.config.num_attention_heads
num_key_value_heads = internlm_model.config.num_key_value_heads
num_key_value_groups = internlm_model.config.num_attention_heads // num_key_value_heads

for name, param in internlm_model.named_parameters():
    if "model.layers." in name:
        layer_str = name.split(".")[2]
        name_for_mapping = name.replace(f".{layer_str}.", f".0.")
        if name_for_mapping in intern_llama_mapping:
            mapped_name = intern_llama_mapping[name_for_mapping].replace(".0.", f".{layer_str}.")
            if 'attention.wqkv.weight' in name:
                # QKV weight 需要特殊处理
                hidden_size = llama_model.config.hidden_size
                
                q_weight, k_weight, v_weight = preproces_internlm_qkvweight(param, num_key_value_groups, head_dim)
                kv_dim = hidden_size//llama_model.config.num_attention_heads *llama_model.config.num_key_value_heads
                mapped_q_name = mapped_name.replace("qkv_proj.weight", "q_proj.weight")
                llama_model.state_dict()[mapped_q_name].copy_(q_weight)
                print(f"mapped_q_name: {mapped_q_name}, shape: {q_weight.shape}")
                mapped_k_name = mapped_name.replace("qkv_proj.weight", "k_proj.weight")
                llama_model.state_dict()[mapped_k_name].copy_(k_weight)
                print(f"mapped_k_name: {mapped_k_name}, shape: {k_weight.shape}")
                mapped_v_name = mapped_name.replace("qkv_proj.weight", "v_proj.weight")
                llama_model.state_dict()[mapped_v_name].copy_(v_weight)
                print(f"mapped_v_name: {mapped_v_name}, shape: {v_weight.shape}")
            else:
                llama_model.state_dict()[mapped_name].copy_(param)
                print(f"mapped_name: {mapped_name}, shape: {param.shape}")
        else:
            print(f"Layer {name} not found in LlamaForCausalLM.")
    elif name in intern_llama_mapping:
        llama_model.state_dict()[intern_llama_mapping[name]].copy_(param)
    else:
        print(f"Layer {name} not found in LlamaForCausalLM.")
print("Finish the conversion of internLM weights to LlaMA!")

# 保存转换后的模型
print("Saving the converted LlaMA...")
llama_model.save_pretrained("internlm_converted_llama")


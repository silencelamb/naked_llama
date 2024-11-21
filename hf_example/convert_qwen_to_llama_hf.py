from transformers import LlamaConfig, AutoModelForCausalLM, LlamaForCausalLM
import torch

# 加载 Qwen2 模型
print("Loading Qwen2 weights...")
with torch.no_grad():
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
print("Loaded Qwen2 weights!")

# 创建 LLaMA2 格式的配置
llama_config = LlamaConfig(
    vocab_size=qwen_model.config.vocab_size,
    hidden_size=qwen_model.config.hidden_size,
    intermediate_size=qwen_model.config.intermediate_size,
    num_hidden_layers=qwen_model.config.num_hidden_layers,
    num_attention_heads=qwen_model.config.num_attention_heads,
    num_key_value_heads=qwen_model.config.num_key_value_heads,
    max_position_embeddings=qwen_model.config.max_position_embeddings,
    rope_theta=qwen_model.config.rope_theta,
    rms_norm_eps=qwen_model.config.rms_norm_eps,
    attention_bias=True, # Qwen2-7B-Instruct 有 Q K V的bias,但是又没有O的bias,所以后面还需要单独对O的bias进行置0处理
    torch_dtype=qwen_model.config.torch_dtype,
    bos_token_id=qwen_model.config.bos_token_id,
    eos_token_id=qwen_model.config.eos_token_id,
)

# 转换模型权重
print("Initializing LlaMA using config from Qwen2...")
with torch.no_grad():
    llama_model = LlamaForCausalLM(llama_config).to(dtype=torch.bfloat16).to(device=qwen_model.device)

# 复制生成配置，这个挺重要的，因为生成配置会影响到模型的输出对齐
# 关键配置比如是否sample，是否repetition_penalty等
llama_model.generation_config = qwen_model.generation_config

# 复制权重(这里需要根据具体层的对应关系进行修改)
print("Start converting Qwen2 weights to LlaMA...")
for name, param in qwen_model.named_parameters():
    if name in llama_model.state_dict():
        llama_model.state_dict()[name].copy_(param)
    else:
        print(f"Layer {name} not found in LlamaForCausalLM.")

# 将 self_attn.o_proj.bias 设置为 0，因为没办法单独设置o的biase为False
# 其实可以修改下config单独配置o的bias，这样就不用手动设置了，也不会有额外计算
for name, param in llama_model.named_parameters():
    if 'self_attn.o_proj.bias' in name:
        print(f'set {name} to zero')
        llama_model.state_dict()[name].copy_(torch.zeros_like(param))
print("Finish the conversion of Qwen2 weights to LlaMA!")

# 保存转换后的模型
print("Saving the converted LlaMA...")
llama_model.save_pretrained("qwen2_converted_llama")

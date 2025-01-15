from transformers import AutoModelForCausalLM, Qwen2ForCausalLM, Qwen2Config, AutoTokenizer
import torch

# 加载 InternLM3-8B 模型
print("Loading internLM weights...")
with torch.no_grad():
    internlm_model = AutoModelForCausalLM.from_pretrained(
        "internlm/internlm3-8b-instruct",
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True
    )
print("Loaded internLM weights!")

# 创建 Qwen2 格式的配置
qwen2_config = Qwen2Config(
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
print("Initializing Qwen2 using config from InternLM...")
with torch.no_grad():
    qwen2_model = Qwen2ForCausalLM(qwen2_config).to(dtype=torch.bfloat16).to(device=internlm_model.device)

# 复制生成配置，这个挺重要的，因为生成配置会影响到模型的输出对齐
# 关键配置比如是否sample，是否repetition_penalty等
qwen2_model.generation_config = internlm_model.generation_config

# 复制权重(这里需要根据具体层的对应关系进行修改)
print("Start converting internLM weights to Qwen2...")


for name, param in internlm_model.named_parameters():
    if name in qwen2_model.state_dict():
        qwen2_model.state_dict()[name].copy_(param)
    else:
        print(f"Layer {name} not found in Qwen2ForCausalLM.")
        
print("Finish the conversion of internLM weights to Qwen2!")

# 保存转换后的模型
print("Saving the converted Qwen2...")
qwen2_model.save_pretrained("internlm3_converted_qwen2")

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm3-8b-instruct", trust_remote_code=True)
tokenizer.save_pretrained("internlm3_converted_qwen2")


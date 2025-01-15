from transformers import LlamaConfig, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer
import torch

# 加载 internlm3-8b 模型
print("Loading InternLM3 weights...")
with torch.no_grad():
    internlm_model = AutoModelForCausalLM.from_pretrained(
        "internlm/internlm3-8b-instruct",
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True
    )
print("Loaded InternLM3 weights!")

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
    rope_scaling=internlm_model.config.rope_scaling,
    rms_norm_eps=internlm_model.config.rms_norm_eps,
    torch_dtype=internlm_model.config.torch_dtype,
    bos_token_id=internlm_model.config.bos_token_id,
    eos_token_id=internlm_model.config.eos_token_id,
)

# 转换模型权重
print("Initializing LlaMA using config from internlm3...")
with torch.no_grad():
    llama_model = LlamaForCausalLM(llama_config).to(dtype=torch.bfloat16).to(device=internlm_model.device)

# 复制生成配置，这个挺重要的，因为生成配置会影响到模型的输出对齐
# 关键配置比如是否sample，是否repetition_penalty等
llama_model.generation_config = internlm_model.generation_config

# 复制权重(这里需要根据具体层的对应关系进行修改)
print("Start converting internlm3 weights to LlaMA...")
for name, param in internlm_model.named_parameters():
    if name in llama_model.state_dict():
        llama_model.state_dict()[name].copy_(param)
    else:
        print(f"Layer {name} not found in LlamaForCausalLM.")

print("Finish the conversion of internlm3 weights to LlaMA!")

# 保存转换后的模型
print("Saving the converted LlaMA...")
llama_model.save_pretrained("internlm3_converted_llama")

tokenizer = AutoTokenizer.from_pretrained("internlm/internlm3-8b-instruct", trust_remote_code=True)
tokenizer.save_pretrained("internlm3_converted_llama")

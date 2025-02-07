from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

save_directory = "DeepSeek-R1-Small-2layers"
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1', trust_remote_code=True)
tokenizer.save_pretrained(save_directory)

# 创建新的配置
config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-R1', trust_remote_code=True)
config.first_k_dense_replace = 1  # Dense层只保留第一个
config.num_hidden_layers = 2  # 一共2层
config.num_nextn_predict_layers = 0
del config.quantization_config

config.intermediate_size = 1024

config.n_routed_experts = 64

config.num_experts_per_tok = 4

config.moe_intermediate_size = 128

# 用新配置初始化模型
with torch.no_grad():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.bfloat16).cuda()
    model.eval()
    model.save_pretrained(save_directory)


    prompt = "Who are u?"
    messages = []
    messages.append({"role": "user", "content": prompt})
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(prompt_tokens, max_new_tokens=100, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_tokens, generated_ids)
    ]
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(completion)
    messages.append({"role": "assistant", "content": completion})
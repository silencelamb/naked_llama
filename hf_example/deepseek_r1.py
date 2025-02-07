from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

model_dir = "DeepSeek-R1-Small-2layers"
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModelForCausalLM.from_pretrained('WisdomShell/CodeShell-7B-Chat', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained('WisdomShell/CodeShell-7B-Chat')

import pdb; pdb.set_trace()
history = []
query = '你是谁?'
response = model.chat(query, history, tokenizer)
print(response)
history.append((query, response))

query = '用Python写一个HTTP server'
response = model.chat(query, history, tokenizer)
print(response)
history.append((query, response))
import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import torch.distributed as dist

"""
Usage:
```bash
EP_SIZE=4 torchrun --nproc_per_node=4 deepseek_r1_ep.py
```
"""

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ep_size = int(os.environ.get("EP_SIZE", 1))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    model_dir = "DeepSeek-R1-Small-2layers"

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if hasattr(config, "ep_size"):
        config.ep_size = ep_size

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(torch.device("cuda", local_rank))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    prompt = "Who are u?"
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(torch.device("cuda", local_rank))
    generated_ids = model.generate(prompt_tokens, max_new_tokens=100, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(prompt_tokens, generated_ids)]
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if local_rank == 0:
        print(completion)

if __name__ == "__main__":
    main()
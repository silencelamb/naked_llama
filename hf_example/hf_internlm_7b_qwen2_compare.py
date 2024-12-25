from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
from typing import List, Optional, Tuple, Union
import torch
from transformers.generation.streamers import BaseStreamer

# cuda is not exactly the same, because different result of cuda gemm kernel, detail example see
# hf_example/merge_qkv.py
# device = "cuda" 
device = "cpu" # cpu is exacatly the same
attn_impl = 'eager' # the attention implementation to use
meta_instruction = ("You are an AI assistant whose name is InternLM (书生·浦语).\n"
"- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory "
"(上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
"- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such "
"as English and 中文."
)
prompt1 = "介绍下你自己"
prompt2 = "介绍下上海人工智能实验室"
print(meta_instruction)

def build_inputs(tokenizer, query: str, history: List[Tuple[str, str]] = None, meta_instruction=meta_instruction):
    if history is None:
        history = []
    if tokenizer.add_bos_token:
        prompt = ""
    else:
        prompt = tokenizer.bos_token
    if meta_instruction:
        prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    for record in history:
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
    return tokenizer([prompt], return_tensors="pt")

@torch.inference_mode()
def chat(
    model: Union[AutoModelForCausalLM, Qwen2ForCausalLM],
    tokenizer,
    query: str,
    history: Optional[List[Tuple[str, str]]] = None,
    streamer: Optional[BaseStreamer] = None,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.8,
    meta_instruction: str = meta_instruction,
    **kwargs,
):
    if history is None:
        history = []
    inputs = build_inputs(tokenizer, query, history, meta_instruction)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if torch.is_tensor(v)}
    # also add end-of-assistant token in eos token id to avoid unnecessary generation
    eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]
    outputs = model.generate(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        **kwargs,
    )
    outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    response = response.split("<|im_end|>")[0]
    history = history + [(query, response)]
    return response, history
    

###############1. 官方demo###############
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2_5-7b-chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("internlm/internlm2_5-7b-chat", 
                                             torch_dtype=torch.bfloat16, 
                                             attn_implementation=attn_impl,
                                             trust_remote_code=True).to(device)
model = model.eval()
response_official_1, history = model.chat(tokenizer, prompt1, history=[], do_sample=False)
print("###############1. 官方demo###############")
print(f"User Input: {prompt1}\nInternLM2.5 Response: {response_official_1}")
response_official_2, history = model.chat(tokenizer, prompt2, history=history, do_sample=False)
print(f"User Input: {prompt2}\nInternLM2.5 Response: {response_official_2}")

###############2. 使用拆分出来的函数后###############
print("###############2. 使用拆分出来的函数后###############")
response_splitfunc_1, history = chat(model, tokenizer, prompt1, history=[], do_sample=False)
print(f"User Input: {prompt1}\nInternLM2.5 Response: {response_splitfunc_1}")
response_splitfunc_2, history = chat(model, tokenizer, prompt2, history=history, do_sample=False)
print(f"User Input: {prompt2}\nInternLM2.5 Response: {response_splitfunc_2}")
print(f'response1 equal: {response_official_1 == response_splitfunc_1}')
print(f'response2 equal: {response_official_2 == response_splitfunc_2}')

###############3. 使用Converted Qwen2 model + 分离出来的function###############
tokenizer = AutoTokenizer.from_pretrained("internlm_converted_qwen2", trust_remote_code=True)
llama_model = Qwen2ForCausalLM.from_pretrained(
    "internlm_converted_qwen2",
    torch_dtype='auto',
    attn_implementation=attn_impl).to(model.device)
llama_model.eval()
response_llama_and_splitfunc_1, history = chat(llama_model, tokenizer, prompt1, history=[], do_sample=False)
print("###############3. 使用Converted Qwen2 model + 分离出来的function###############")
print(f"User Input: {prompt1}\nConverted Qwen2 Response: {response_llama_and_splitfunc_1}")

response_llama_and_splitfunc_2, history = chat(llama_model, tokenizer, prompt2, history=history, do_sample=False)
print(f"User Input: {prompt2}\nConverted Qwen2 Response: {response_llama_and_splitfunc_2}")
print(f'response1 equal: {response_official_1 == response_llama_and_splitfunc_1}')
print(f'response2 equal: {response_official_2 == response_llama_and_splitfunc_2}')

###############4. logits比对###############
print("###############4. logits比对###############")

model_inputs = build_inputs(tokenizer, prompt1).to(model.device)
# # detail result to compare logits
with torch.no_grad():
    detail_output = model(input_ids = model_inputs.input_ids, output_hidden_states=True, output_attentions=True, return_dict = True)
    llama_detail_output = llama_model(input_ids = model_inputs.input_ids, output_hidden_states=True, output_attentions=True, return_dict = True)

    print(f'logits equal: {(detail_output["logits"] == llama_detail_output["logits"]).all()}')
    print(torch.allclose(llama_detail_output['logits'].float(), detail_output['logits']))
    print((llama_detail_output['hidden_states'][0] == detail_output['hidden_states'][0]).all())
    print((llama_detail_output['hidden_states'][1] == detail_output['hidden_states'][1]).all())
    print(torch.allclose(llama_detail_output['hidden_states'][1], detail_output['hidden_states'][1]))
    print((llama_detail_output['attentions'][0] == detail_output['attentions'][0]).all())
    print(torch.allclose(llama_detail_output['attentions'][0], detail_output['attentions'][0]))
    
    
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
device = "cuda" # the device to load the model onto
attn_impl = 'eager' # the attention implementation to use

prompt = "大模型和人工智能经历了两年的快速发展，请你以此主题对人工智能的从业者写一段新年寄语"

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."""
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt},
 ]

tokenizer = AutoTokenizer.from_pretrained("internlm3_converted_llama", trust_remote_code=True)
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

print(prompt)

# Official InternLM model
internlm_model = AutoModelForCausalLM.from_pretrained(
    "internlm/internlm3-8b-instruct", 
    torch_dtype='auto',
    attn_implementation=attn_impl,
    trust_remote_code=True).cuda()
internlm_generated_ids = internlm_model.generate(model_inputs.input_ids, max_new_tokens=100, do_sample=False)
internlm_generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, internlm_generated_ids)
]
print('Original InernLM3-8B-Instruct:')
# print(internlm_generated_ids)
internlm_response = tokenizer.batch_decode(internlm_generated_ids, skip_special_tokens=True)[0]
print(internlm_response)

# Converted LlaMA model
llama_model = LlamaForCausalLM.from_pretrained(
    "internlm3_converted_llama",
    torch_dtype='auto',
    attn_implementation=attn_impl).cuda()
llama_generated_ids = llama_model.generate(model_inputs.input_ids, max_new_tokens=100, do_sample=False)
llama_generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, llama_generated_ids)
]

# detail result to compare logits
internlm_detail_output = internlm_model(input_ids = model_inputs.input_ids, output_hidden_states = True, return_dict = True)
llama_detail_output = llama_model(input_ids = model_inputs.input_ids, output_hidden_states = True, return_dict = True)

print('Converted Llama:')
# print(llama_generated_ids)
llama_response = tokenizer.batch_decode(llama_generated_ids, skip_special_tokens=True)[0]
print(llama_response)

print(f'total generated {len(internlm_generated_ids[0])} tokens')
print(f'generated_ids equal: {(internlm_generated_ids[0] == llama_generated_ids[0]).all()}')
print(f'response equal: {internlm_response == llama_response}')

print(f'logits equal: {(internlm_detail_output["logits"] == llama_detail_output["logits"]).all()}')

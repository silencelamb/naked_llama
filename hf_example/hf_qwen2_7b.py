from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
device = "cuda" # the device to load the model onto
attn_impl = 'eager' # the attention implementation to use

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

# Qwen model
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct", 
    torch_dtype='auto',
    attn_implementation=attn_impl).cuda()
qwen_generated_ids = qwen_model.generate(model_inputs.input_ids, max_new_tokens=32, do_sample=False)
qwen_generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, qwen_generated_ids)
]
print('Original Qwen2-7B-Instruct:')
print(qwen_generated_ids)
qwen_response = tokenizer.batch_decode(qwen_generated_ids, skip_special_tokens=True)[0]
print(qwen_response)

# Converted LlaMA model
llama_model = LlamaForCausalLM.from_pretrained(
    "qwen2_converted_llama",
    torch_dtype='auto',
    attn_implementation=attn_impl).cuda()
llama_generated_ids = llama_model.generate(model_inputs.input_ids, max_new_tokens=32, do_sample=False)
llama_generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, llama_generated_ids)
]

# detail result to compare logits
qwen_detail_output = qwen_model(input_ids = model_inputs.input_ids, output_hidden_states = True, return_dict = True)
llama_detail_output = llama_model(input_ids = model_inputs.input_ids, output_hidden_states = True, return_dict = True)

print('Converted Llama:')
print(llama_generated_ids)
llama_response = tokenizer.batch_decode(llama_generated_ids, skip_special_tokens=True)[0]
print(llama_response)

print(f'total generated {len(qwen_generated_ids[0])} tokens')
print(f'generated_ids equal: {(qwen_generated_ids[0] == llama_generated_ids[0]).all()}')
print(f'response equal: {qwen_response == llama_response}')

print(f'logits equal: {(qwen_detail_output["logits"] == llama_detail_output["logits"]).all()}')

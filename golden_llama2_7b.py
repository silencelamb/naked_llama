import torch
from transformers import AutoTokenizer, LlamaForCausalLM



if __name__ == '__main__':

    
    # tokenization
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids
    # check result
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    cpu_res = model(input_ids = token_ids,
                    output_attentions = True,
                    output_hidden_states = True,
                    return_dict = True
                    )
    import pdb; pdb.set_trace()
    
    

    
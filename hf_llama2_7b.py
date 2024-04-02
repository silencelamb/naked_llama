import torch
from transformers import AutoTokenizer, LlamaForCausalLM, TextStreamer
from transformers import set_seed
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F
from enum import Enum


PROMPT_FORWARD = 0b000001             # 1、 一次性prompt的forward
HF_GENERATE = 0b0000010               # 2、 HF's model.generate
HF_STREAM_GENERATE = 0b0000100        # 3、 流式generate，model.generate传入TextStreamer，实现有word结果立即显示
FORWARD = 0b000001000                 # 4、 自己写的，每次只生成一个token，每次都是新的token与已有序列拼到一起，然后forward
FORWARD_KV_CACHE = 0b00010000         # 5、 在 4的基础上，利用kv cache，每次只需输入新产生的token做prompt
FORWARD_HF_CACHE = 0b00100000         # 6、 跟5类似，但是使用HF的最新的KV cache Instance


ModeSwitch = PROMPT_FORWARD | HF_GENERATE | HF_STREAM_GENERATE | FORWARD | FORWARD_KV_CACHE | FORWARD_HF_CACHE

if __name__ == '__main__':
    
    # tokenization
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    prompt_str = "Hey, are you conscious? Can you talk to me?"
    max_gen_len = 30
    inputs = tokenizer(prompt_str, return_tensors="pt")
    token_ids = inputs.input_ids
    # check result
    model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    with torch.inference_mode():
        ########### 1、promt forward ####################
        if PROMPT_FORWARD & ModeSwitch:
            cpu_res = model(input_ids = token_ids,
                            output_attentions = True,
                            output_hidden_states = True,
                            return_dict = True
                            )
        
        ########### 2、Generate ##########################
        # https://huggingface.co/docs/transformers/en/model_doc/llama2#transformers.LlamaForCausalLM.forward.example
        if HF_GENERATE & ModeSwitch:
            print('########### 2、Generate ######################')
            set_seed(65536)
            generate_ids = model.generate(inputs.input_ids, max_length=max_gen_len)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(output)
            
        ########## 3、Stream Generate ###################
        # https://huggingface.co/docs/transformers/v4.39.2/en/internal/generation_utils#transformers.TextStreamer
        if HF_STREAM_GENERATE & ModeSwitch:
            print('########### 3、Stream Generate ###################')
            set_seed(65536)
            streamer = TextStreamer(tokenizer)
            generate_ids = model.generate(inputs.input_ids, streamer=streamer, max_length=max_gen_len)


        ########### 4、use model forward###################
        # https://github.com/mit-han-lab/streaming-llm/blob/main/examples/run_streaming_llama.py#L19
        if FORWARD & ModeSwitch:
            print('########### 4、use model forward###################')
            set_seed(65536)
            gen_length = 0
            prompt = token_ids
            while gen_length < max_gen_len:
                output = model(input_ids=prompt)
                # obtain next token
                # output.logits.shape is:  [batch_size, seq_len, vocab_size]
                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                # add the next token to prompt
                prompt = torch.cat((prompt, pred_token_idx), dim=1)
                next_word = tokenizer.batch_decode(pred_token_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                print(next_word[0])
                gen_length += 1
            whole_response = tokenizer.batch_decode(prompt, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f'whole_response: \n{whole_response}')
        
        ########### 5、model forward with KV cache###################
        # https://github.com/mit-han-lab/streaming-llm/blob/main/examples/run_streaming_llama.py#L19
        if FORWARD_KV_CACHE & ModeSwitch:
            print('########### 5、model forward with KV cache###################')
            set_seed(65536)
            gen_length = 0
            past_key_values = None
            total_pred_tokens = torch.tensor([[]], dtype=torch.long)
            prompt = token_ids
            while gen_length < max_gen_len:
                output = model(input_ids=prompt, past_key_values=past_key_values, use_cache=True)
                # update kv cache
                past_key_values = output.past_key_values
                # obtain next token
                # output.logits.shape is:  [batch_size, seq_len, vocab_size]
                probs = F.softmax(output.logits[:, -1, :], dim=-1)
                max_probs, pred_token_idx = probs.max(dim=-1)
                pred_token_idx = pred_token_idx.unsqueeze(1)
                next_word = tokenizer.batch_decode(pred_token_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                total_pred_tokens = torch.cat((total_pred_tokens, pred_token_idx), dim=1)
                print(f'next token: id {pred_token_idx[0].item()}, {next_word[0]}, prob: {max_probs.item(): 0.4f}')
                # only use the new token as the prompt
                prompt = pred_token_idx
                gen_length += 1
            whole_response = tokenizer.batch_decode(total_pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f'whole_response: \n{prompt_str}{whole_response}')

        ########### 6、model forward with hf new KV cache Instance ###################
        # https://github.com/huggingface/transformers/pull/26681
        # https://huggingface.co/docs/transformers/en/model_doc/llama2#transformers.LlamaForCausalLM.forward.past_key_values
        # https://huggingface.co/docs/transformers/v4.39.2/en/internal/generation_utils#transformers.DynamicCache
        # pip install transformers==4.39.2
        if FORWARD_HF_CACHE & ModeSwitch:
            print('########### 6、model forward with hf new KV cache Instance ###################')
            set_seed(65536)
            gen_length = 0
            kv_cache = DynamicCache()
            total_pred_tokens = torch.tensor([[]], dtype=torch.long)
            prompt = token_ids
            while gen_length < max_gen_len:
                output = model(input_ids=prompt, past_key_values=kv_cache, use_cache=True)
                # obtain next token
                probs = F.softmax(output.logits[:, -1, :], dim=-1)
                max_probs, pred_token_idx = probs.max(dim=-1)
                pred_token_idx = pred_token_idx.unsqueeze(1)
                next_word = tokenizer.batch_decode(pred_token_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                total_pred_tokens = torch.cat((total_pred_tokens, pred_token_idx), dim=1)
                print(f'next token: id {pred_token_idx[0].item()}, {next_word[0]}, prob: {max_probs.item(): 0.4f}')
                # only use the new token as the prompt
                prompt = pred_token_idx
                gen_length += 1
            whole_response = tokenizer.batch_decode(total_pred_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            print(f'whole_response: \n{prompt_str}{whole_response}')
            
from transformers import LlamaTokenizer

if __name__ == '__main__':
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    # text = '你好'
    text = 'What a nice day today'
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.encode(text,return_tensors="pt")
    print('tokens:', tokens)
    print('input_ids:', input_ids)
    
    # tokens: ['▁What', '▁a', '▁nice', '▁day', '▁today', '.']
    # input_ids: tensor([[    1,  1724,   263,  7575,  2462,  9826, 29889]])
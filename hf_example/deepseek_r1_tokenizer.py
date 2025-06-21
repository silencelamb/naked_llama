from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-R1', trust_remote_code=True)

# 定义消息列表
messages = [
    # {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "Can you tell me a joke?"},
    {"role": "assistant", "content": "bla bla bla, 讲完了, 好不好笑"},
    {"role": "user", "content": "一点也不好笑。。。 还是帮我写代码吧"},
]

# 应用 chat template，add_generation_prompt=True 表示最后会加 assistant 的回答提示
chat_string = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False  # 返回字符串而不是 token ids
)
print(chat_string)
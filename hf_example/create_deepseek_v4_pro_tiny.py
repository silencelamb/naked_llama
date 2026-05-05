from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
import torch

save_directory = "DeepSeek-V4-Pro-Tiny"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load tokenizer from DeepSeek V4 Pro
print("Downloading tokenizer from deepseek-ai/DeepSeek-V4-Pro...")
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V4-Pro', trust_remote_code=True)
tokenizer.save_pretrained(save_directory)
print("Tokenizer saved.")

# Load config from official DeepSeek V4 Pro
print("Loading config from deepseek-ai/DeepSeek-V4-Pro...")
config = AutoConfig.from_pretrained('deepseek-ai/DeepSeek-V4-Pro', trust_remote_code=True)

# Modify config: only change layers and experts, keep everything else
config.num_hidden_layers = 6
config.n_routed_experts = 32
config.hidden_size = 500
config.moe_intermediate_size = 300

config.compress_ratios = [128, 128, 4, 128, 4, 128, 0]  # 6 layers + 1 nextn
config.layer_types = [
    "heavily_compressed_attention",
    "heavily_compressed_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
]
config.mlp_layer_types = ["hash_moe", "hash_moe", "hash_moe", "moe", "moe", "moe"]
del config.quantization_config
del config.expert_dtype

print(f"num_hidden_layers: {config.num_hidden_layers}")
print(f"n_routed_experts: {config.n_routed_experts}")
print(f"hidden_size: {config.hidden_size}")

# Initialize model with random weights
print("Initializing model with random weights...")
with torch.no_grad():
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).to(torch.bfloat16).to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_size_gb = total_params * 2 / (1024**3)  # bfloat16 = 2 bytes
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (bfloat16): {total_size_gb:.2f} GB")

    # Save model
    print(f"Saving model to {save_directory}...")
    model.save_pretrained(save_directory)
    print("Model saved.")

    # Test inference
    print("\nTesting inference...")
    prompt = "Who are you?"
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(prompt_tokens, max_new_tokens=50, do_sample=False)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(prompt_tokens, generated_ids)
    ]
    completion = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Prompt: {prompt}")
    print(f"Completion: {completion}")
    print("\nDone!")

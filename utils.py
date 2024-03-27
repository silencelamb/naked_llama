import numpy as np
import torch
import json
from configuration_llama import LlamaConfig


def npy_to_tensor(npy_name):
    # 将 NumPy 数组转换回 PyTorch 张量
    loaded_numpy_array = np.load(npy_name)
    loaded_tensor = torch.from_numpy(loaded_numpy_array)
    loaded_tensor = loaded_tensor.to(torch.float32) # 将张量转换为 float32 类型
    return loaded_tensor


def load_llama_config(config_file):
    with open(config_file, "r") as f:
        config_data = json.load(f)
    configuration = LlamaConfig(**config_data)
    return configuration


def get_attentioin_mask(start_pos, seq_length, ref_tensor):
    if seq_length > 1:
        mask = torch.full((seq_length, seq_length), float("-inf"), device=ref_tensor.device)
        
        mask = torch.triu(mask, diagonal=1)
        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        mask = torch.hstack([
            torch.zeros((seq_length, start_pos), device=ref_tensor.device),
            mask
        ]).type_as(ref_tensor)
    else:
        mask = None
    return mask
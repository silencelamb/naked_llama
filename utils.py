import numpy as np
import torch


def npy_to_tensor(npy_name):
    # 将 NumPy 数组转换回 PyTorch 张量
    loaded_numpy_array = np.load(npy_name)
    loaded_tensor = torch.from_numpy(loaded_numpy_array)
    return loaded_tensor

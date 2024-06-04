import torch
import torch_xla
import torch_xla.core.xla_model as xm
import sys
sys.path.append("..")
from utils import get_attentioin_mask
import os

# OPTIMIZED=True
OPTIMIZED=False

if OPTIMIZED:
    os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "true"
else:
    os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "false"


# 创建一个随机初始化的张量
dev = xm.xla_device()
rand_tensor = torch.randn(2,2).to(dev)

# 调用函数
mask = get_attentioin_mask(3, 5, rand_tensor)

# HLO IR
hlo = torch_xla._XLAC._get_xla_tensors_hlo([mask])
print(hlo)

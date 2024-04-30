import torch
import torch_xla
import torch_xla.core.xla_model as xm
import sys
sys.path.append("..")
from layers.attention import repeat_kv
import os

OPTIMIZED=True
# OPTIMIZED=False

if OPTIMIZED:
    os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "true"
else:
    os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "false"


dev = xm.xla_device()

batch_size, num_kv_heads, sequence_length, head_dimension = 1, 2, 4, 3
n_rep = 2  # 重复次数

# 创建一个随机初始化的隐藏状态张量
hidden_states = torch.randn(batch_size, num_kv_heads, sequence_length, head_dimension)
hidden_states = hidden_states.to(dev)

# 调用函数
output = repeat_kv(hidden_states, n_rep)

# aten IR
print(torch_xla._XLAC._get_xla_tensors_text([output]))

# HLO IR
hlo = torch_xla._XLAC._get_xla_tensors_hlo([output])
print(hlo)

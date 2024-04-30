import torch
import torch_xla
import torch_xla.core.xla_model as xm
import sys
sys.path.append("..")
from layers.attention import repeat_kv
import os

os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "true"
os.environ["XLA_SAVE_TENSORS_FMT"] = "hlo"
os.environ["XLA_SAVE_TENSORS_FILE"] = "./dump_mlir/repeat_kv.hlo"
os.environ['XLA_DUMP_HLO_GRAPH'] = "true"

dev = xm.xla_device()

batch_size, num_kv_heads, sequence_length, head_dimension = 1, 2, 4, 3
n_rep = 2  # 重复次数

# 创建一个随机初始化的隐藏状态张量
hidden_states = torch.randn(batch_size, num_kv_heads, sequence_length, head_dimension)
hidden_states = hidden_states.to(dev)

# 调用函数
output = repeat_kv(hidden_states, n_rep)

xm.mark_step()

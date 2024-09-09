import torch
import torch_xla
import torch_xla.core.xla_model as xm
import sys
sys.path.append("..")
from layers.matmul import MLP
import os

os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "true"
os.environ["XLA_SAVE_TENSORS_FMT"] = "hlo"
os.environ["XLA_SAVE_TENSORS_FILE"] = "./dump_mlir/matmul.hlo"
os.environ['XLA_DUMP_HLO_GRAPH'] = "true"

batch_size, seq_len, hidden_size, intermediate_size = 16, 1024, 4096, 11008
# 创建一个随机初始化的张量
dev = xm.xla_device()

# 假设输入是一个需要梯度的张量
input_tensor = torch.randn(batch_size, seq_len, hidden_size).requires_grad_(True).to(dev)
# 定义权重
weight = torch.ones(intermediate_size, hidden_size).requires_grad_(True).to(dev)
bias = torch.randn(intermediate_size).requires_grad_(True).to(dev)

mlp = MLP(weight, bias)

# 前向传递
output_class = mlp.forward(input_tensor)

# 定义输出的grad
dy = .1 * torch.randn_like(output_class)

output_class.backward(dy, retain_graph=True)

xm.mark_step()

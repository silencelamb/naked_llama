import torch
import torch_xla
import torch_xla.core.xla_model as xm
import sys
sys.path.append("..")
from utils import get_attentioin_mask
import os

os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "true"
os.environ["XLA_SAVE_TENSORS_FMT"] = "hlo"
os.environ["XLA_SAVE_TENSORS_FILE"] = "./dump_mlir/casual_mask.hlo"
os.environ['XLA_DUMP_HLO_GRAPH'] = "true"

# 创建一个随机初始化的张量
dev = xm.xla_device()
rand_tensor = torch.randn(2,2).to(dev)

# 调用函数
mask = get_attentioin_mask(3, 5, rand_tensor)

xm.mark_step()

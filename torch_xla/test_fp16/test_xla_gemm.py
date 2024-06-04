import torch
import torch_xla.core.xla_model as xm
device = xm.xla_device()
a = torch.randn(3, 3, dtype=torch.float16)
b = torch.randn(3, 3, dtype=torch.float16)
a = a.to(device)
b = b.to(device)
c = torch.matmul(a, b)
print(c)

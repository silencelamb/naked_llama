import torch


# 创建 FP16 张量
a = torch.randn(3, 3, dtype=torch.float16)
b = torch.randn(3, 3, dtype=torch.float16)

# 矩阵乘法
c = torch.matmul(a, b)

print(c)

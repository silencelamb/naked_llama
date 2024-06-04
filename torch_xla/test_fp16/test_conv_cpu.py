import torch
import torch.nn as nn

# 创建一个简单的卷积网络
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = ConvNet()

# 创建 FP16 输入张量
input_tensor = torch.randn(1, 1, 28, 28, dtype=torch.float16)

# 前向传播
output = model(input_tensor)

print(output)

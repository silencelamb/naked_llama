import torch
from torch import nn
import torch.testing


def silu_backward(dy, x):
    # origin from xformers : https://github.com/facebookresearch/xformers/blob/main/xformers/ops/swiglu_op.py#L83
    # xformers refers: https://github.com/pytorch/pytorch/blob/563b065f5a4b4055fa6b025c2514b566d5fd9439/aten/src/ATen/native/Activation.cpp#L483
    sigm = 1 / (1 + torch.exp(-x.float()))
    return (dy.float() * sigm * (1 + x.float() * (1 - sigm))).to(x.dtype)


class Softmax:
    def __init__(self, dim=-1):
        self.dim = dim
        self.cache = None
    
    def forward(self, x, dtype=torch.float32):
        # softmax_rst = torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True) 
        # 使用稳定实现的softmax
        softmax_rst = nn.functional.softmax(x, dim=self.dim, dtype=dtype)
        self.cache = softmax_rst
        return softmax_rst

    def backward(self, grad_output):
        softmax_rst = self.cache
        grad_x = softmax_rst * (grad_output - torch.sum(grad_output * softmax_rst, dim=-1, keepdim=True))
        return grad_x

def test_silu_backward_manual_func():
    # silu反向计算比较：手写的反向实现与pytorch自带的自动求导
    eps = 1e-5
    batch_size = 4
    seq_len = 12
    feature_size = 128

    # 假设输入是一个需要梯度的张量

    input_tensor = torch.randn(batch_size, seq_len, feature_size)
    # 定义输出的grad
    dy = .1 * torch.randn_like(input_tensor)
    
    input_tensor.requires_grad_(True)

    # 前向传递
    output_ref = nn.functional.silu(input_tensor)
    # reference 的backward结果
    output_ref.backward(dy, retain_graph=True)
    dx_ref = input_tensor.grad.clone()
    # manual backward的结果
    dx_manual = silu_backward(dy, input_tensor)

    print(torch.testing.assert_close(dx_ref, dx_manual))
    
if __name__ == "__main__":
    test_silu_backward_manual_func()

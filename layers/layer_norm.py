import torch
from torch import nn
import torch.testing

class LayerNorm:
    # from https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.py
    # you can also refer https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    def __init__(self, weight, bias, eps=1e-5):
        self.eps = eps
        self.weight = weight
        self.bias = bias
        self.cache = None
        
    def forward(self, x):
        B, T, C = x.size()
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        rstd = (var + self.eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        out = norm * self.weight + self.bias # B,T,C

        self.cache = (x, mean, rstd)
        return out

    def backward(self, dout):
        x, mean, rstd = self.cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * self.weight
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db


def test_LayerNorm_backward_manual_class():
    # from https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.py
    # 测试手写的LayerNorm反向实现，与torch的自动求导比对
    B = 2
    T = 3
    C = 4
    x = torch.randn(B, T, C, requires_grad=True)
    w = torch.randn(C, requires_grad=True)
    b = torch.randn(C, requires_grad=True)
    layernorm = LayerNorm(w, b)
    out = layernorm.forward(x)

    dout = torch.randn(B, T, C)
    dx, dw, db = layernorm.backward(dout)

    # compare to PyTorch autograd
    fakeloss = (out * dout).sum()
    fakeloss.backward()
    print("dx error:", (x.grad - dx).abs().max().item())
    print("dw error:", (w.grad - dw).abs().max().item())
    print("db error:", (b.grad - db).abs().max().item())

    
if __name__ == "__main__":
    test_LayerNorm_backward_manual_class()
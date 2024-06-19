import torch
from torch import nn
import torch.testing

class LayerNorm:
    # from https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.py
    # you can also refer https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    @staticmethod
    def forward(x, w, b, eps=1e-5):
        B, T, C = x.size()
        mean = x.sum(-1, keepdim=True) / C # B,T,1
        xshift = x - mean # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C # B,T,1
        rstd = (var + eps) ** -0.5 # B,T,1
        norm = xshift * rstd # B,T,C
        out = norm * w + b # B,T,C

        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        # gradients for weights, bias
        db = dout.sum((0, 1))
        dw = (dout * norm).sum((0, 1))
        # gradients for input
        dnorm = dout * w
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)
        dx *= rstd
        return dx, dw, db


def test_LayerNorm_backward_manual_func():
    # from https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.py
    # 测试手写的LayerNorm反向实现，与torch的自动求导比对
    B = 2
    T = 3
    C = 4
    x = torch.randn(B, T, C, requires_grad=True)
    w = torch.randn(C, requires_grad=True)
    b = torch.randn(C, requires_grad=True)
    out, cache = LayerNorm.forward(x, w, b)

    dout = torch.randn(B, T, C)
    dx, dw, db = LayerNorm.backward(dout, cache)

    # compare to PyTorch autograd
    fakeloss = (out * dout).sum()
    fakeloss.backward()
    print("dx error:", (x.grad - dx).abs().max().item())
    print("dw error:", (w.grad - dw).abs().max().item())
    print("db error:", (b.grad - db).abs().max().item())

    
if __name__ == "__main__":
    test_LayerNorm_backward_manual_func()
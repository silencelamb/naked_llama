import torch
import torch.nn as nn

def MLP(x, weight, bias=None):
    # 全连接层/矩阵乘
    if bias is None:
        output = torch.matmul(x, weight)
    else:
        output = torch.matmul(x, weight) + bias
    return output

def FFN_up(x, weight, bias=None):
    return MLP(x, weight, bias)

def FFN_down(x, weight, bias=None):
    return MLP(x, weight, bias)

def FFN_gate(x, weight, bias=None):
    return MLP(x, weight, bias)

def LlamaMLP(hidden_states, w_up, w_gate, w_down):
    
    up_proj = FFN_up(hidden_states, w_up)
    gate_proj = FFN_gate(hidden_states, w_gate)
    gate_proj = nn.functional.silu(gate_proj)
    up_proj = up_proj * gate_proj
    
    down_proj = FFN_down(up_proj, w_down)

    return down_proj

def lm_head(x, weight, bias=None):
    return MLP(x, weight, bias)
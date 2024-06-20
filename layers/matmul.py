import torch
import torch.nn as nn
import torch.testing
from activation import silu_backward

def MLP(x, weight, bias=None):
    # 全连接层/矩阵乘
    if bias is None:
        output = torch.matmul(x, weight.T)
    else:
        output = torch.matmul(x, weight.T) + bias
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

    
class LlamaMLP_origin(nn.Module):
    # from transformers modeling_llama.py
    def __init__(self, hidden_size, intermediate_size, mlp_bias=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.act_fn = nn.functional.silu

    def forward(self, x):
        gate_proj = self.gate_proj(x)
        up_proj = self.up_proj(x)
        activated_gate_proj = self.act_fn(gate_proj)
        intermediate_states = activated_gate_proj * up_proj
        down_proj = self.down_proj(intermediate_states)
        return down_proj


def MLP_backward(grad_output, x, w_up, w_gate, w_down):
    # 手写 MLP 的反向计算
    grad_x = None
    grad_w_up = None
    grad_w_gate = None
    grad_w_down = None

    B, S, H = x.shape

    # 上投影、门控投影和下投影的前向计算
    up_proj = torch.matmul(x, w_up.T)
    gate_proj = torch.matmul(x, w_gate.T)
    gate_proj_silu = torch.nn.functional.silu(gate_proj)
    up_proj_gate = up_proj * gate_proj_silu
    # down_proj = torch.matmul(up_proj_gate, w_down.T)
    
    # 计算门控投影的梯度
    # 基于torch.sigmoid的实现
    # grad_gate_proj = torch.matmul(grad_output, w_down) * up_proj * torch.sigmoid(gate_proj) * (1 + gate_proj * (1 - torch.sigmoid(gate_proj)))
    # 基于./layers/activation.py 中的silu_backward的实现
    grad_gate_proj = torch.matmul(grad_output, w_down) * up_proj * silu_backward(torch.ones_like(gate_proj), gate_proj)

    # 计算上投影的梯度
    # grad_up_proj = grad_output * gate_proj

    # 计算下投影的梯度
    grad_up_proj_gate = torch.matmul(grad_output, w_down) * gate_proj_silu

    # 计算输入的梯度
    if x.requires_grad:
        grad_x = torch.matmul(grad_up_proj_gate, w_up) + torch.matmul(grad_gate_proj, w_gate)

    # 计算权重的梯度
    if w_up.requires_grad:
        grad_w_up = torch.matmul(grad_up_proj_gate.view(B*S, -1).T, x.view(B*S, -1))
    if w_gate.requires_grad:
        grad_w_gate = torch.matmul(grad_gate_proj.view(B*S, -1).T, x.view(B*S, -1))
    if w_down.requires_grad:
        grad_w_down = torch.matmul(grad_output.view(B*S, -1).T, up_proj_gate.view(B*S, -1))

    return grad_x, grad_w_up, grad_w_gate, grad_w_down


def test_MLP_backward_auto_class_and_func():
    # 使用pytorch自带的自动求导
    # MLP的实现：一种使用nn.module的类，一种使用函数
    batch_size = 4
    seq_len = 12
    hidden_size = 128
    intermediate_size = 64
    llama_mlp = LlamaMLP_origin(hidden_size, intermediate_size)

    # 假设输入是一个需要梯度的张量

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    
    # 定义输出的grad
    dy = .1 * torch.randn_like(input_tensor)
    
    input_tensor.requires_grad_(True)

    # 前向传递
    output_class = llama_mlp(input_tensor)
    
    output_class.backward(dy, retain_graph=True)
    dx_class, dg_class, du_class, dd_class = [_.grad.clone() for _ in [input_tensor, 
                                                   llama_mlp.gate_proj.weight,
                                                   llama_mlp.up_proj.weight, 
                                                   llama_mlp.down_proj.weight]]

    weight_g = llama_mlp.gate_proj.weight.clone().detach().requires_grad_(True)
    weight_u = llama_mlp.up_proj.weight.clone().detach().requires_grad_(True)
    weight_d = llama_mlp.down_proj.weight.clone().detach().requires_grad_(True)
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    
    output_func = LlamaMLP(input_tensor, weight_u, weight_g, weight_d)

    output_func.backward(dy, retain_graph=True)
    dx_func, du_func, dg_func, dd_func = [_.grad.clone() for _ in [input_tensor, weight_u, weight_g, weight_d]]
    
    print(torch.testing.assert_close(output_class, output_func))
    print(torch.testing.assert_close(dx_class, dx_func))
    print(torch.testing.assert_close(du_class, du_func))
    print(torch.testing.assert_close(dg_class, dg_func))
    print(torch.testing.assert_close(dd_class, dd_func))


def test_MLP_backward_manual_func():
    # MLP反向计算比较：手写的反向实现与pytorch自带的自动求导
    batch_size = 4
    seq_len = 12
    hidden_size = 128
    intermediate_size = 64
    llama_mlp = LlamaMLP_origin(hidden_size, intermediate_size)

    # 假设输入是一个需要梯度的张量

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    # 定义输出的grad
    dy = .1 * torch.randn_like(input_tensor)
    
    input_tensor.requires_grad_(True)

    # 前向传递
    output_class = llama_mlp(input_tensor)
    
    output_class.backward(dy, retain_graph=True)
    dx_class, dg_class, du_class, dd_class = [_.grad.clone() for _ in [input_tensor, 
                                                   llama_mlp.gate_proj.weight,
                                                   llama_mlp.up_proj.weight, 
                                                   llama_mlp.down_proj.weight]]

    weight_g = llama_mlp.gate_proj.weight.clone().detach().requires_grad_(True)
    weight_u = llama_mlp.up_proj.weight.clone().detach().requires_grad_(True)
    weight_d = llama_mlp.down_proj.weight.clone().detach().requires_grad_(True)
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    dx_manual, du_manual, dg_manual, dd_manual = MLP_backward(dy, input_tensor, weight_u, weight_g, weight_d)

    print(torch.testing.assert_close(dx_class, dx_manual))
    print(torch.testing.assert_close(du_class, du_manual))
    print(torch.testing.assert_close(dg_class, dg_manual))
    print(torch.testing.assert_close(dd_class, dd_manual))


if __name__ == "__main__":
    test_MLP_backward_auto_class_and_func()
    test_MLP_backward_manual_func()




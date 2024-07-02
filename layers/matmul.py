import torch
import torch.nn as nn
import torch.testing
from layers.activation import silu_backward

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

def matmul_backward(grad_output, x, weight, bias=None):

    grad_x = None
    grad_weight = None
    grad_bias = None

    o_z, i_z = weight.shape

    if x.requires_grad:
        grad_x = torch.matmul(grad_output, weight)
    if weight.requires_grad:
        # 张量转置时，需改变形状
        grad_weight = torch.matmul(grad_output.view(-1, o_z).T, x.view(-1, i_z))

    if bias is not None and bias.requires_grad:
        grad_bias = grad_output.sum(dim=0)

        return grad_x, grad_weight, grad_bias

    return grad_x, grad_weight

def LlamaMLP_backward(grad_output, x, w_up, w_gate, w_down):
    # 手写 MLP 的反向计算
    grad_x = None
    grad_w_up = None
    grad_w_gate = None
    grad_w_down = None

    # 前向传播
    up_proj = FFN_up(x, w_up)
    gate_proj = FFN_gate(x, w_gate)
    gate_proj_act = nn.functional.silu(gate_proj)
    up_proj_gate = up_proj * gate_proj_act
    down_proj = FFN_down(up_proj_gate, w_down)

    # 反向传播
    # 下投影梯度
    grad_up_proj_gate, grad_w_down = matmul_backward(grad_output, up_proj_gate, w_down)
    
    # 门控投影和上投影结果的梯度
    grad_up_proj = grad_up_proj_gate * gate_proj_act
    grad_gate_proj_act = grad_up_proj_gate * up_proj
    
    # silu激活函数反向传播
    grad_gate_proj = silu_backward(grad_gate_proj_act, gate_proj)
    
    # 门控投影梯度
    grad_hidden_states_gate, grad_w_gate = matmul_backward(grad_gate_proj, x, w_gate)
    
    # 上投影梯度
    grad_hidden_states_up, grad_w_up = matmul_backward(grad_up_proj, x, w_up)
    
    # 总的输入梯度
    grad_x = grad_hidden_states_gate + grad_hidden_states_up
    
    return grad_x, grad_w_up, grad_w_gate, grad_w_down

def test_matmul_backward_manual_func():
    # MLP反向计算比较：手写的反向实现与pytorch自带的自动求导
    batch_size = 4
    seq_len = 12
    hidden_size = 128
    intermediate_size = 64

    # 假设输入是一个需要梯度的张量
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    # 定义权重
    weight = nn.Parameter(torch.ones(intermediate_size, hidden_size ))
    
    input_tensor.requires_grad_(True)

    # 前向传递
    output_class = MLP(input_tensor, weight)

    # 定义输出的grad
    dy = .1 * torch.randn_like(output_class)
    
    output_class.backward(dy, retain_graph=True)
    dx_class, dw_class = [_.grad.clone() for _ in [input_tensor, weight]]

    weight = weight.clone().detach().requires_grad_(True)
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    dx_manual, dw_manual = matmul_backward(dy, input_tensor, weight)

    print(torch.testing.assert_close(dx_class, dx_manual))
    print(torch.testing.assert_close(dw_class, dw_manual))

def test_LlamaMLP_backward_auto_class_and_func():
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
    dx_class, du_class, dg_class, dd_class = [_.grad.clone() for _ in [input_tensor, 
                                                   llama_mlp.up_proj.weight,
                                                   llama_mlp.gate_proj.weight, 
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


def test_LlamaMLP_backward_manual_func():
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
    dx_class, du_class, dg_class, dd_class = [_.grad.clone() for _ in [input_tensor, 
                                                   llama_mlp.up_proj.weight,
                                                   llama_mlp.gate_proj.weight, 
                                                   llama_mlp.down_proj.weight]]

    weight_g = llama_mlp.gate_proj.weight.clone().detach().requires_grad_(True)
    weight_u = llama_mlp.up_proj.weight.clone().detach().requires_grad_(True)
    weight_d = llama_mlp.down_proj.weight.clone().detach().requires_grad_(True)
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    dx_manual, du_manual, dg_manual, dd_manual = LlamaMLP_backward(dy, input_tensor, weight_u, weight_g, weight_d)

    print(torch.testing.assert_close(dx_class, dx_manual))
    print(torch.testing.assert_close(du_class, du_manual))
    print(torch.testing.assert_close(dg_class, dg_manual))
    print(torch.testing.assert_close(dd_class, dd_manual))


if __name__ == "__main__":
    test_LlamaMLP_backward_auto_class_and_func()
    test_LlamaMLP_backward_manual_func()
    test_matmul_backward_manual_func()




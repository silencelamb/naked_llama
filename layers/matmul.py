import torch
import torch.nn as nn
import torch.testing
from layers.activation import silu_backward

class MLP:
    # 全连接层/矩阵乘
    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias
        self.cache = None
    
    def forward(self, x):
        if self.bias is None:
            output = torch.matmul(x, self.weight.T)
        else:
            output = torch.matmul(x, self.weight.T) + self.bias
        self.cache = x
        return output

    def backward(self, grad_output):
        grad_x, grad_weight, grad_bias = None, None, None
        x = self.cache
        o_z, i_z = self.weight.shape

        if x.requires_grad:
            grad_x = torch.matmul(grad_output, self.weight)
        if self.weight.requires_grad:
            # 张量转置时，需改变形状
            grad_weight = torch.matmul(grad_output.view(-1, o_z).T, x.view(-1, i_z))

        if self.bias is not None and self.bias.requires_grad:
            grad_bias = grad_output.sum(dim=0)
            while len(grad_bias.shape) > len(self.bias.shape):
                grad_bias = grad_bias.sum(dim=0)
            return grad_x, grad_weight, grad_bias

        return grad_x, grad_weight, grad_bias


class LoraMLP:
    # Lora的全连接层/矩阵乘
    def __init__(self, base_weight, lora_a, lora_b, scaling, dropout, base_bias=None):
        base_weight.require_grad_(False)
        if base_bias is not None:
            base_bias.require_grad_(False)
        self.base_linear = MLP(base_weight, base_bias)
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scaling = scaling
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.cache = None
    
    def eval(self):
        self.dropout.eval()
    
    def train(self):
        self.dropout.train()
    
    def forward(self, x):
        result = self.base_linear.forward(x)
        dropout_x = self.dropout(x)
        lora_branch_a = torch.matmul(dropout_x, self.lora_a.T)
        lora_branch_b = torch.matmul(lora_branch_a, self.lora_b.T)
        lora_branch = self.scaling * lora_branch_b
        result += lora_branch
        self.cache = (x, dropout_x, lora_branch_a,)
        return result

    def backward(self, grad_output):
        grad_x_base, grad_weight_base, grad_bias_base = self.base_linear.backward(grad_output)
        
        assert grad_weight_base is None and grad_bias_base is None, "Base weight and bias should not have gradients"
        
        x, dropout_x, lora_branch_a = self.cache
        grad_lora_a, grad_lora_b = None, None

        r_z, i_z = self.lora_a.shape
        o_z, _ = self.lora_b.shape
        # 计算 Lora 分支的梯度
        grad_lora_temp = torch.matmul(grad_output, self.lora_b) * self.scaling

        if x.requires_grad:
            mask = (dropout_x != 0).float()
            grad_x_lora = torch.matmul(grad_lora_temp, self.lora_a) 
            grad_x_lora = grad_x_lora * mask / (1 - self.dropout_rate) 
            grad_x = grad_x_base + grad_x_lora 
        
        if self.lora_a.requires_grad:
            grad_lora_a = torch.matmul(grad_lora_temp.view(-1, r_z).T, dropout_x.view(-1, i_z))
        
        if self.lora_b.requires_grad:
            grad_lora_b = torch.matmul(grad_output.view(-1, o_z).T, lora_branch_a.view(-1, r_z)) * self.scaling
        
        return grad_x, grad_lora_a, grad_lora_b

class LlamaMLP:
    def __init__(self, w_up, w_gate, w_down, bias_up=None, bias_gate=None, bias_down=None):
        self.FFN_up = MLP(w_up, bias_up)
        self.FFN_gate = MLP(w_gate, bias_gate)
        self.FFN_down = MLP(w_down, bias_down)
        self.cache = None
    
    def forward(self, hidden_states):
        up_proj = self.FFN_up.forward(hidden_states)
        gate_proj = self.FFN_gate.forward(hidden_states)
        gate_proj_act = nn.functional.silu(gate_proj)
        up_proj_gate = up_proj * gate_proj_act
        down_proj = self.FFN_down.forward(up_proj_gate)
        self.cache = (up_proj, gate_proj, gate_proj_act)
        return down_proj

    def backward(self, grad_output):
        # 手写 LlamaMLP 的反向计算
        grad_x = None
        grad_w_up, grad_bias_up = None, None
        grad_w_gate, grad_bias_gate = None, None
        grad_w_down, grad_bias_down = None, None
        up_proj, gate_proj, gate_proj_act = self.cache

        # 下投影梯度
        grad_up_proj_gate, grad_w_down, grad_bias_down = self.FFN_down.backward(grad_output)
        
        # 门控投影和上投影结果的梯度
        grad_up_proj = grad_up_proj_gate * gate_proj_act
        grad_gate_proj_act = grad_up_proj_gate * up_proj
        
        # silu激活函数反向传播
        grad_gate_proj = silu_backward(grad_gate_proj_act, gate_proj)
        
        # 门控投影梯度
        grad_hidden_states_gate, grad_w_gate, grad_bias_gate = self.FFN_gate.backward(grad_gate_proj)
        
        # 上投影梯度
        grad_hidden_states_up, grad_w_up, grad_bias_up = self.FFN_up.backward(grad_up_proj)
        
        # 总的输入梯度
        grad_x = grad_hidden_states_gate + grad_hidden_states_up
        
        return grad_x, grad_w_up, grad_w_gate, grad_w_down, grad_bias_up, grad_bias_gate, grad_bias_down

class LoraLlamaMLP(LlamaMLP):
    
    def replace_with_lora(self, up_lora_a, up_lora_b, gate_lora_a, gate_lora_b, down_lora_a, down_lora_b, scaling, dropout):
        self.FFN_up.weight.requires_grad_(False)
        self.FFN_gate.weight.requires_grad_(False)
        self.FFN_down.weight.requires_grad_(False)
        self.FFN_up = LoraMLP(self.FFN_up.weight, up_lora_a, up_lora_b, scaling, dropout, self.FFN_up.bias)
        self.FFN_gate = LoraMLP(self.FFN_gate.weight, gate_lora_a, gate_lora_b, scaling, dropout, self.FFN_gate.bias)
        self.FFN_down = LoraMLP(self.FFN_down.weight, down_lora_a, down_lora_b, scaling, dropout, self.FFN_down.bias)

    def eval(self):
        self.FFN_up.eval()
        self.FFN_gate.eval()
        self.FFN_down.eval()
    
    def train(self):
        self.FFN_up.train()
        self.FFN_gate.train()
        self.FFN_down.train()
    
    def backward(self, grad_output):
        grad_x = None
        up_proj, gate_proj, gate_proj_act = self.cache
        # 反向梯度计算类似 LlamaMLP
        grad_up_proj_gate, grad_lora_a_down, grad_lora_b_down = self.FFN_down.backward(grad_output)
        
        grad_up_proj = grad_up_proj_gate * gate_proj_act
        grad_gate_proj_act = grad_up_proj_gate * up_proj
        
        grad_gate_proj = silu_backward(grad_gate_proj_act, gate_proj)
        
        grad_hidden_states_gate, grad_lora_a_gate, grad_lora_b_gate = self.FFN_gate.backward(grad_gate_proj)
        
        grad_hidden_states_up, grad_lora_a_up, grad_lora_b_up = self.FFN_up.backward(grad_up_proj)
        
        grad_x = grad_hidden_states_gate + grad_hidden_states_up
        
        return grad_x, grad_lora_a_up, grad_lora_b_up, grad_lora_a_gate, grad_lora_b_gate, grad_lora_a_down, grad_lora_b_down
    
class LlamaMLPOrigin(nn.Module):
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

def test_matmul_backward_manual_class():
    # MLP反向计算比较：手写的反向实现与pytorch自带的自动求导
    batch_size = 4
    seq_len = 12
    hidden_size = 128
    intermediate_size = 64

    # 假设输入是一个需要梯度的张量
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    # 定义权重
    weight = torch.randn(intermediate_size, hidden_size).requires_grad_(True)
    bias = torch.randn(intermediate_size).requires_grad_(True)
    
    input_tensor.requires_grad_(True)
    mlp = MLP(weight, bias)

    # 前向传递
    output_class = mlp.forward(input_tensor)

    # 定义输出的grad
    dy = .1 * torch.randn_like(output_class)
    
    output_class.backward(dy, retain_graph=True)
    dx_class, dw_class, dbias_class = [_.grad.clone() for _ in [input_tensor, weight, bias]]

    dx_manual, dw_manual, dbias_manual = mlp.backward(dy)

    print(torch.testing.assert_close(dx_class, dx_manual, rtol=1e-20, atol=1e-20))
    print(torch.testing.assert_close(dw_class, dw_manual))
    print(torch.testing.assert_close(dbias_class, dbias_manual))


def test_LlamaMLP_backward_manual_class():
    # MLP反向计算比较：手写的反向实现与pytorch自带的自动求导
    batch_size = 4
    seq_len = 12
    hidden_size = 128
    intermediate_size = 64
    # bias = True, only for test, llama2 is without bias, however Qwen2 is with bias
    llama_mlp_origin = LlamaMLPOrigin(hidden_size, intermediate_size, mlp_bias=True)

    # 假设输入是一个需要梯度的张量

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    # 定义输出的grad
    dy = .1 * torch.randn_like(input_tensor)
    
    input_tensor.requires_grad_(True)

    # 前向传递
    output_class = llama_mlp_origin(input_tensor)
    
    output_class.backward(dy, retain_graph=True)
    dx_class, du_class, dg_class, dd_class, dbias_u_class, dbias_g_class, dbias_d_class = \
                                                [_.grad.clone() for _ in [input_tensor, 
                                                   llama_mlp_origin.up_proj.weight,
                                                   llama_mlp_origin.gate_proj.weight, 
                                                   llama_mlp_origin.down_proj.weight,
                                                   llama_mlp_origin.up_proj.bias,
                                                   llama_mlp_origin.gate_proj.bias, 
                                                   llama_mlp_origin.down_proj.bias]]

    weight_g = llama_mlp_origin.gate_proj.weight.clone().detach().requires_grad_(True)
    bias_g = llama_mlp_origin.gate_proj.bias.clone().detach().requires_grad_(True)
    weight_u = llama_mlp_origin.up_proj.weight.clone().detach().requires_grad_(True)
    bias_u = llama_mlp_origin.up_proj.bias.clone().detach().requires_grad_(True)
    weight_d = llama_mlp_origin.down_proj.weight.clone().detach().requires_grad_(True)
    bias_d = llama_mlp_origin.down_proj.bias.clone().detach().requires_grad_(True)
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    llama_mlp_manual = LlamaMLP(weight_u, weight_g, weight_d, bias_u, bias_g, bias_d)
    output_manual = llama_mlp_manual.forward(input_tensor)
    dx_manual, du_manual, dg_manual, dd_manual, dbias_u_manual, dbias_g_manual, dbias_d_manual = llama_mlp_manual.backward(dy)

    print(torch.testing.assert_close(output_class, output_manual))
    print(torch.testing.assert_close(dx_class, dx_manual, rtol=1e-8, atol=1e-8))
    print(torch.testing.assert_close(du_class, du_manual, rtol=1e-20, atol=1e-20))
    print(torch.testing.assert_close(dg_class, dg_manual, rtol=1e-7, atol=1e-7))
    print(torch.testing.assert_close(dd_class, dd_manual, rtol=1e-20, atol=1e-20))
    print(torch.testing.assert_close(dbias_u_class, dbias_u_manual))
    print(torch.testing.assert_close(dbias_g_class, dbias_g_manual))
    print(torch.testing.assert_close(dbias_d_class, dbias_d_manual))

def test_LlamaMLP_backward_manual_auto():
    # MLP反向计算比较：手写的反向实现与pytorch自带的自动求导
    batch_size = 4
    seq_len = 12
    hidden_size = 128
    intermediate_size = 64
    # bias = True, only for test, llama2 is without bias, however Qwen2 is with bias
    weight_u = torch.randn(intermediate_size, hidden_size).requires_grad_(True)
    weight_g = torch.randn(intermediate_size, hidden_size).requires_grad_(True)
    weight_d = torch.randn(hidden_size, intermediate_size).requires_grad_(True)
    bias_u = torch.randn(intermediate_size).requires_grad_(True)
    bias_g = torch.randn(intermediate_size).requires_grad_(True)
    bias_d = torch.randn(hidden_size).requires_grad_(True)

    llama_mlp = LlamaMLP(weight_u, weight_g, weight_d, bias_u, bias_g, bias_d)

    # 假设输入是一个需要梯度的张量

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    # 定义输出的grad
    dy = .1 * torch.randn_like(input_tensor)
    
    input_tensor.requires_grad_(True)

    # 前向传递
    output_class = llama_mlp.forward(input_tensor)
    
    output_class.backward(dy, retain_graph=True)
    dx_class, du_class, dg_class, dd_class, dbias_u_class, dbias_g_class, dbias_d_class = \
                                                [_.grad.clone() for _ in [input_tensor, 
                                                   llama_mlp.FFN_up.weight,
                                                   llama_mlp.FFN_gate.weight, 
                                                   llama_mlp.FFN_down.weight,
                                                   llama_mlp.FFN_up.bias,
                                                   llama_mlp.FFN_gate.bias, 
                                                   llama_mlp.FFN_down.bias]
                                                 ]

    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    dx_manual, du_manual, dg_manual, dd_manual, dbias_u_manual, dbias_g_manual, dbias_d_manual = llama_mlp.backward(dy)

    print(torch.testing.assert_close(dx_class, dx_manual, rtol=1e-10, atol=1e-10))
    print(torch.testing.assert_close(du_class, du_manual))
    print(torch.testing.assert_close(dg_class, dg_manual))
    print(torch.testing.assert_close(dd_class, dd_manual))
    print(torch.testing.assert_close(dbias_u_class, dbias_u_manual))
    print(torch.testing.assert_close(dbias_g_class, dbias_g_manual))
    print(torch.testing.assert_close(dbias_d_class, dbias_d_manual))

def test_LoraMLP_backward_manual_class():
    # LoraMLP反向计算比较：手写的反向实现与PyTorch自带的自动求导
    batch_size = 2
    seq_len = 6
    hidden_size = 64
    intermediate_size = 8
    dropout_rate = 0.1
    scaling_factor = 0.5

    # 初始化LoraMLP和相关参数
    base_weight = torch.randn(hidden_size, hidden_size, requires_grad=False)
    lora_a = torch.randn(intermediate_size, hidden_size, requires_grad=True)
    lora_b = torch.randn(hidden_size, intermediate_size, requires_grad=True)
    base_bias = torch.randn(hidden_size, requires_grad=True)
    dropout = dropout_rate

    lora_mlp = LoraMLP(base_weight, lora_a, lora_b, scaling_factor, dropout, base_bias)

    # 假设输入是一个需要梯度的张量
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    dy = 0.1 * torch.randn_like(input_tensor)

    # 前向传递
    output = lora_mlp.forward(input_tensor)

    # 自动求导反向传播
    output.backward(dy, retain_graph=True)
    dx_class, dlora_a_class, dlora_b_class, = [_.grad.clone() for _ in [input_tensor, lora_a, lora_b]]

    # 手写反向传播
    dx_manual, dlora_a_manual, dlora_b_manual = lora_mlp.backward(dy)

    gradient_names = ["x", "lora_a", "lora_b", ]
    auto_grads = [dx_class,  dlora_a_class, dlora_b_class,]
    manual_grads = [dx_manual, dlora_a_manual, dlora_b_manual,]

    # 比较反向传播的梯度
    for name, auto_grad, manual_grad in zip(gradient_names, auto_grads, manual_grads):
        try:
            torch.testing.assert_close(auto_grad, manual_grad)
            print(f"Gradient match for {name}: PASSED")
        except AssertionError as e:
            print(f"Gradient match for {name}: FAILED")
            print(e)

def test_LoraLlamaMLP_backward_manual_class():
    # LoraLlamaMLP反向计算比较：手写的反向实现与PyTorch自带的自动求导
    batch_size = 2
    seq_len = 6
    hidden_size = 64
    intermediate_size = 8
    dropout_rate = 0.1
    scaling_factor = 0.5

    # 初始化LoraLlamaMLP和相关参数
    w_up = torch.randn(hidden_size, hidden_size, requires_grad=False)
    w_gate = torch.randn(hidden_size, hidden_size, requires_grad=False)
    w_down = torch.randn(hidden_size, hidden_size, requires_grad=False)
    bias_up = torch.randn(hidden_size, requires_grad=False)
    bias_gate = torch.randn(hidden_size, requires_grad=False)
    bias_down = torch.randn(hidden_size, requires_grad=False)

    up_lora_a = torch.randn(intermediate_size, hidden_size, requires_grad=True)
    up_lora_b = torch.randn(hidden_size, intermediate_size, requires_grad=True)
    gate_lora_a = torch.randn(intermediate_size, hidden_size, requires_grad=True)
    gate_lora_b = torch.randn(hidden_size, intermediate_size, requires_grad=True)
    down_lora_a = torch.randn(intermediate_size, hidden_size, requires_grad=True)
    down_lora_b = torch.randn(hidden_size, intermediate_size, requires_grad=True)
    dropout = dropout_rate

    lora_llama_mlp = LoraLlamaMLP(w_up, w_gate, w_down, bias_up, bias_gate, bias_down)
    lora_llama_mlp.replace_with_lora(up_lora_a, up_lora_b, gate_lora_a, gate_lora_b, down_lora_a, down_lora_b, scaling_factor, dropout)


    # 假设输入是一个需要梯度的张量
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    dy = 0.1 * torch.randn_like(input_tensor)

    # 前向传递
    output = lora_llama_mlp.forward(input_tensor)

    # 自动求导反向传播
    output.backward(dy, retain_graph=True)
    dx_class, dlora_a_up_class, dlora_b_up_class, \
            dlora_a_gate_class, dlora_b_gate_class, dlora_a_down_class, dlora_b_down_class = \
        [_.grad.clone() for _ in [input_tensor, up_lora_a, up_lora_b, gate_lora_a, \
                              gate_lora_b, down_lora_a, down_lora_b]]

    # 手写反向传播
    dx_manual, dlora_a_up_manual, dlora_b_up_manual, dlora_a_gate_manual, \
        dlora_b_gate_manual, dlora_a_down_manual, dlora_b_down_manual = lora_llama_mlp.backward(dy)

    gradient_names = ["x", "lora_a_up", "lora_b_up", "lora_a_gate", "lora_b_gate",\
                          "lora_a_down", "lora_b_down"]
    auto_grads = [dx_class, dlora_a_up_class, dlora_b_up_class, \
                    dlora_a_gate_class, dlora_b_gate_class, dlora_a_down_class, dlora_b_down_class]
    manual_grads = [dx_manual, dlora_a_up_manual, dlora_b_up_manual, \
                        dlora_a_gate_manual, dlora_b_gate_manual, dlora_a_down_manual, dlora_b_down_manual]

    # 比较反向传播的梯度
    for name, auto_grad, manual_grad in zip(gradient_names, auto_grads, manual_grads):
        try:
            torch.testing.assert_close(auto_grad, manual_grad, rtol=5e-5, atol=5e-5)
            # torch.testing.assert_close(auto_grad, manual_grad)

            print(f"Gradient match for {name}: PASSED")
        except AssertionError as e:
            print(f"Gradient match for {name}: FAILED")
            print(e)


if __name__ == "__main__":
    # test_matmul_backward_manual_class()
    test_LlamaMLP_backward_manual_class()
    # test_LlamaMLP_backward_manual_auto()
    # test_LoraMLP_backward_manual_class()
    test_LoraLlamaMLP_backward_manual_class()




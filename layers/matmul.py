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
        self.base_linear = MLP(base_weight, base_bias)
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scaling = scaling
        self.dropout = nn.Dropout(p=dropout)
        self.cache = None
    
    def eval(self):
        self.dropout.eval()
    
    def train(self):
        self.dropout.train()
    
    def forward(self, x):
        result = self.base_linear.forward(x)
        dropout_x = self.dropout(x)
        lora_branch = torch.matmul( torch.matmul(dropout_x, self.lora_a.T), self.lora_b.T )
        lora_branch = self.scaling * lora_branch
        result += lora_branch
        self.cache = x
        return result

    def backward(self, grad_output):
        pass

        return grad_x, grad_weight, grad_bias

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
        pass
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
    weight = torch.ones(intermediate_size, hidden_size).requires_grad_(True)
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

    print(torch.testing.assert_close(dx_class, dx_manual))
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
                                                   llama_mlp_origin.down_proj.bias]
                                                 ]

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
    print(torch.testing.assert_close(dx_class, dx_manual))
    print(torch.testing.assert_close(du_class, du_manual))
    print(torch.testing.assert_close(dg_class, dg_manual))
    print(torch.testing.assert_close(dd_class, dd_manual))
    print(torch.testing.assert_close(dbias_u_class, dbias_u_manual))
    print(torch.testing.assert_close(dbias_g_class, dbias_g_manual))
    print(torch.testing.assert_close(dbias_d_class, dbias_d_manual))



if __name__ == "__main__":
    test_matmul_backward_manual_class()
    test_LlamaMLP_backward_manual_class()
    




import torch
from torch import nn
import torch.testing

class LlamaRMSNormOrigin(nn.Module):
    # from transformers modeling_llama.py
    def __init__(self, hidden_size, eps=1e-5):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaRMSNorm:
    def __init__(self, weight, eps=1e-5):
        self.weight = weight
        self.variance_epsilon = eps
        self.cache = None
        
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        inv_rms = torch.rsqrt(variance + self.variance_epsilon)
        self.cache = (hidden_states, inv_rms)
        hidden_states = hidden_states * inv_rms
        return self.weight * hidden_states.to(input_dtype)
    
    def backward(self, grad_output):
        # 手写 rmsnorm的反向计算
        # reference: https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rms_layernorm.py
        x, inv_rms = self.cache
        D = x.size(-1)
        grad_x = None
        grad_weight = None

        if x.requires_grad:
            # 使用计算的逆均方根值进行梯度的计算
            grad_x_part1 = grad_output * self.weight * inv_rms
            grad_x_part2 = (x * (grad_output * self.weight) * inv_rms.pow(3)).mean(-1, keepdim=True) * x
            grad_x = grad_x_part1 - grad_x_part2

            
        if self.weight.requires_grad:
            # weight梯度
            grad_weight = (grad_output * x * inv_rms)
            if grad_weight.dim() == 3:
                grad_weight = grad_weight.sum(dim=0).sum(dim=0)
            elif grad_weight.dim() == 2:
                grad_weight = grad_weight.sum(dim=0)

        return grad_x, grad_weight

def test_RMSNorm_backward_manual_class():
    # RMSNorm反向计算比较：手写的class与pytorch自带的自动求导
    eps = 1e-5
    batch_size = 4
    seq_len = 12
    feature_size = 128
    llama_rmsnorm = LlamaRMSNormOrigin(feature_size, eps=eps)

    # 假设输入是一个需要梯度的张量

    input_tensor = torch.randn(batch_size, seq_len, feature_size)
    # 定义输出的grad
    dy = .1 * torch.randn_like(input_tensor)
    
    input_tensor.requires_grad_(True)

    # 前向传递

    output_class = llama_rmsnorm(input_tensor)
    
    output_class.backward(dy, retain_graph=True)
    dx_class, dw_class = [_.grad.clone() for _ in [input_tensor, llama_rmsnorm.weight]]

    # 查看梯度
    # print("Gradients on the input tensor:", dx_class)
    # print("Gradients on the weight parameter:", dw_class)

    weight = llama_rmsnorm.weight.clone().detach().requires_grad_(True)
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    llama_rmsnorm_manual = LlamaRMSNorm(weight, eps=eps)
    output_manual = llama_rmsnorm_manual.forward(input_tensor)
    dx_manual, dw_manual = llama_rmsnorm_manual.backward(dy)

    print(torch.testing.assert_close(dw_class, dw_manual))
    print(torch.testing.assert_close(dx_class, dx_manual))
    
if __name__ == "__main__":
    test_RMSNorm_backward_manual_class()
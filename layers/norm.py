import torch
from torch import nn


def RMSNorm(hidden_states, weight, eps=1e-5):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    hidden_states = weight * hidden_states.to(input_dtype)
    return hidden_states
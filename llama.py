import os.path as osp
import torch
from torch import nn
from layers.embedding import Embedding
from layers.transformer_block import LlamaTransformerBlock, LoraLlamaTransformerBlock
from layers.rms_norm import LlamaRMSNorm
from layers.matmul import MLP 
from layers.rope import init_rope_embeddings
from layers.loss import CrossEntropy
from utils import npy_to_tensor, load_llama_config, get_attentioin_mask
from configuration_llama import LlamaConfig
import torch.nn.functional as F
from transformers import AutoTokenizer, LlamaForCausalLM


class LlamaModel():
    def __init__(self, config: LlamaConfig):
        self.config = config
        embdding_weights = npy_to_tensor(osp.join(config.weights_dir, 'model.embed_tokens.weight.npy'))
        self.embdding = Embedding(embdding_weights.requires_grad_(True))
        
        self.transformer_blocks = []
        for i in range(config.num_hidden_layers):
            w_q = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.weight.npy'))
            w_k = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.weight.npy'))
            w_v = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.weight.npy'))
            w_o = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.weight.npy'))
            input_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.input_layernorm.weight.npy'))
            post_att_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.post_attention_layernorm.weight.npy'))
            w_up = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.up_proj.weight.npy'))
            w_gate = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.gate_proj.weight.npy'))
            w_down = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.down_proj.weight.npy'))
            bias_q_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.bias.npy')
            bias_q = npy_to_tensor(bias_q_file) if osp.exists(bias_q_file) else None
            bias_k_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.bias.npy')
            bias_k = npy_to_tensor(bias_k_file) if osp.exists(bias_k_file) else None
            bias_v_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.bias.npy')
            bias_v = npy_to_tensor(bias_v_file) if osp.exists(bias_v_file) else None
            bias_o_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.bias.npy')
            bias_o = npy_to_tensor(bias_o_file) if osp.exists(bias_o_file) else None
            transformer_block = LlamaTransformerBlock(config, input_norm_weight.requires_grad_(True), 
                                                      w_q.requires_grad_(True), w_k.requires_grad_(True), 
                                                      w_v.requires_grad_(True), w_o.requires_grad_(True), 
                                                      w_up.requires_grad_(True), w_gate.requires_grad_(True), 
                                                      w_down.requires_grad_(True), post_att_norm_weight.requires_grad_(True), 
                                                      bias_q, bias_k, bias_v, bias_o)
            self.transformer_blocks.append(transformer_block)
        
        norm_weight = npy_to_tensor(osp.join(config.weights_dir, 'model.norm.weight.npy'))
        self.output_rmsnorm = LlamaRMSNorm(norm_weight.requires_grad_(True), config.rms_norm_eps)
        
        lm_head_weight = npy_to_tensor(osp.join(config.weights_dir, 'lm_head.weight.npy'))
        self.lm_head = MLP(lm_head_weight.requires_grad_(True))
        
        self.cache = None
    
    def forward(self, token_ids: torch.Tensor):
        """
        手动实现llama2 7B/13B/70B的推理计算。
        
        参数:
        - token_ids: token id组成的tensor，形状为 [batch_size, seq_length]
        """
        bsz, seq_length = token_ids.shape
        # embedding 
        
        input_embeds = self.embdding.forward(token_ids)
        hidden_states = input_embeds  # shape [batch_size, seq_length, hidden_size], hidden_size=4096
        
        # mask
        mask = get_attentioin_mask(start_pos=0, seq_length=seq_length, ref_tensor=hidden_states)

        # 重复 32次(7B)/ 80次(70B) llama2_transformer_block 的计算
        
        for layer_id, transformer_block in enumerate(self.transformer_blocks):
            print(f'Naked llama: Computing {self.config.model_name} Layer {layer_id}')
            hidden_states = transformer_block.forward(hidden_states, attention_mask=mask)
        
        # 先 RMSNorm，然后head输出
        hidden_states = self.output_rmsnorm.forward(hidden_states)
        logits = self.lm_head.forward(hidden_states)        

        return logits
    
    def backward(self, grad_output):
        """
        llama2反向
        """
        grads = {}
        
        # 反向传播 lm_head 
        grad_hs, grad_lm_head_weight, _ = self.lm_head.backward(grad_output)
        grads['lm_head_weight_grad'] = grad_lm_head_weight.clone()

        # 反向传播 RMSNorm
        grad_hs, grad_norm_weight = self.output_rmsnorm.backward(grad_hs)
        grads['final_norm_weight_grad'] = grad_norm_weight.clone()
        
        # 反向传播 Transformer blocks
        for layer_id in reversed(range(len(self.transformer_blocks))):
            grad_output = self.transformer_blocks[layer_id].backward(grad_hs)
            grad_hs = grad_output[0]

            # 存储每个 transformer block 的梯度
            grads[f'block_{layer_id}_input_layernorm_weight_grad'] = grad_output[1].clone()
            grads[f'block_{layer_id}_self_attn_q_proj_weight_grad'] = grad_output[2].clone()
            grads[f'block_{layer_id}_self_attn_k_proj_weight_grad'] = grad_output[3].clone()
            grads[f'block_{layer_id}_self_attn_v_proj_weight_grad'] = grad_output[4].clone()
            grads[f'block_{layer_id}_self_attn_o_proj_weight_grad'] = grad_output[5].clone()
            grads[f'block_{layer_id}_post_attention_layernorm_weight_grad'] = grad_output[6].clone()
            grads[f'block_{layer_id}_mlp_up_proj_weight_grad'] = grad_output[7].clone()
            grads[f'block_{layer_id}_mlp_gate_proj_weight_grad'] = grad_output[8].clone()
            grads[f'block_{layer_id}_mlp_down_proj_weight_grad'] = grad_output[9].clone()
            if grad_output[10] is not None:
                grads[f'block_{layer_id}_self_attn_q_proj_bias_grad'] = grad_output[10].clone()
            if grad_output[11] is not None:
                grads[f'block_{layer_id}_self_attn_k_proj_bias_grad'] = grad_output[11].clone()
            if grad_output[12] is not None:
                grads[f'block_{layer_id}_self_attn_v_proj_bias_grad'] = grad_output[12].clone()
            if grad_output[13] is not None:
                grads[f'block_{layer_id}_self_attn_o_proj_bias_grad'] = grad_output[13].clone()

        
        # 反向传播 embedding 
        grad_embedding_weights = self.embdding.backward(grad_hs)
        grads['embedding_weight_grad'] = grad_embedding_weights.clone()
        
        return grads



class LoraLlamaModel(LlamaModel):
    def __init__(self, config: LlamaConfig):
        super(LoraLlamaModel, self).__init__(config)
        for i in range(config.num_hidden_layers):
            w_q = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.weight.npy'))
            w_k = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.weight.npy'))
            w_v = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.weight.npy'))
            w_o = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.weight.npy'))
            input_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.input_layernorm.weight.npy'))
            post_att_norm_weight = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.post_attention_layernorm.weight.npy'))
            w_up = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.up_proj.weight.npy'))
            w_gate = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.gate_proj.weight.npy'))
            w_down = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.down_proj.weight.npy'))
            bias_q_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.bias.npy')
            bias_q = npy_to_tensor(bias_q_file) if osp.exists(bias_q_file) else None
            bias_k_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.bias.npy')
            bias_k = npy_to_tensor(bias_k_file) if osp.exists(bias_k_file) else None
            bias_v_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.bias.npy')
            bias_v = npy_to_tensor(bias_v_file) if osp.exists(bias_v_file) else None
            bias_o_file = osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.bias.npy')
            bias_o = npy_to_tensor(bias_o_file) if osp.exists(bias_o_file) else None
            
            self.transformer_blocks[i] = LoraLlamaTransformerBlock(config, input_norm_weight, w_q, w_k, w_v, w_o, w_up, \
                w_gate, w_down, post_att_norm_weight, bias_q, bias_k, bias_v, bias_o)
        
    def replace_with_lora(self, config: LlamaConfig, scaling, dropout):
        for i, transformer_block in enumerate(self.transformer_blocks):
            q_lora_a = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.lora_A.weight.npy'))
            q_lora_b = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.q_proj.lora_B.weight.npy'))
            k_lora_a = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.lora_A.weight.npy'))
            k_lora_b = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.k_proj.lora_B.weight.npy'))
            v_lora_a = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.lora_A.weight.npy'))
            v_lora_b = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.v_proj.lora_B.weight.npy'))
            o_lora_a = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.lora_A.weight.npy'))
            o_lora_b = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.self_attn.o_proj.lora_B.weight.npy'))
            up_lora_a = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.up_proj.lora_A.weight.npy'))
            up_lora_b = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.up_proj.lora_B.weight.npy'))
            gate_lora_a = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.gate_proj.lora_A.weight.npy'))
            gate_lora_b = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.gate_proj.lora_B.weight.npy'))
            down_lora_a = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.down_proj.lora_A.weight.npy'))
            down_lora_b = npy_to_tensor(osp.join(config.weights_dir, f'model.layers.{i}.mlp.down_proj.lora_B.weight.npy'))
            
            transformer_block.replace_with_lora(q_lora_a, q_lora_b, k_lora_a, k_lora_b, v_lora_a, v_lora_b, o_lora_a, \
                o_lora_b, up_lora_a, up_lora_b, gate_lora_a, gate_lora_b, down_lora_a, down_lora_b, scaling, dropout)
    
    def eval(self):
        for i, transformer_block in enumerate(self.transformer_blocks):
            transformer_block.eval()
    
    def train(self):
        for i, transformer_block in enumerate(self.transformer_blocks):
            transformer_block.train()
        
    
    def backward(self, grad_output):
        """
        lora_llama2反向
        """
        lora_grads = {}
        
        # 反向传播 lm_head 
        grad_hs, _, _ = self.lm_head.backward(grad_output)

        # 反向传播 RMSNorm
        grad_hs, _ = self.output_rmsnorm.backward(grad_hs)
        
        # 反向传播 Transformer blocks
        for layer_id in reversed(range(len(self.transformer_blocks))):
            grad_output = self.transformer_blocks[layer_id].backward(grad_hs)
            grad_hs = grad_output[0]

            # 存储每个 transformer block 的梯度
            
            lora_grads[f'block_{layer_id}_self_attn.q_proj.lora_A_grad'] = grad_output[1].clone()
            lora_grads[f'block_{layer_id}_self_attn.q_proj.lora_B_grad'] = grad_output[2].clone()
            lora_grads[f'block_{layer_id}_self_attn.k_proj.lora_A_grad'] = grad_output[3].clone()
            lora_grads[f'block_{layer_id}_self_attn.k_proj.lora_B_grad'] = grad_output[4].clone()
            lora_grads[f'block_{layer_id}_self_attn.v_proj.lora_A_grad'] = grad_output[5].clone()
            lora_grads[f'block_{layer_id}_self_attn.v_proj.lora_B_grad'] = grad_output[6].clone()
            lora_grads[f'block_{layer_id}_self_attn.o_proj.lora_A_grad'] = grad_output[7].clone()
            lora_grads[f'block_{layer_id}_self_attn.o_proj.lora_B_grad'] = grad_output[8].clone()

            lora_grads[f'block_{layer_id}_mlp.up_proj.lora_A_grad'] = grad_output[9].clone()
            lora_grads[f'block_{layer_id}_mlp.up_proj.lora_B_grad'] = grad_output[10].clone()
            lora_grads[f'block_{layer_id}_mlp.gate_proj.lora_A_grad'] = grad_output[11].clone()
            lora_grads[f'block_{layer_id}_mlp.gate_proj.lora_B_grad'] = grad_output[12].clone()
            lora_grads[f'block_{layer_id}_mlp.down_proj.lora_A_grad'] = grad_output[13].clone()
            lora_grads[f'block_{layer_id}_mlp.down_proj.lora_B_grad'] = grad_output[14].clone()
        
        return lora_grads


def test_llama_backward_manual_class():
    # initial rope embeddings
    init_rope_embeddings(dim=128)
    prompt = "Hey, are you conscious? Can you talk to me?"
    model_dict = {
        "llama2_7b": {
            'tokenizer': 'meta-llama/Llama-2-7b-hf',
            'hf_model': 'meta-llama/Llama-2-7b-hf',
            'config_path': 'configs/llama2_7b_config.json',
            'weights_dir': 'weights/llama2_7b/'
        },
    }
    model_name = "llama2_7b"
        
    print('Model:', model_name) 
    config = load_llama_config(model_dict[model_name]['config_path'])
    config.weights_dir = model_dict[model_name]['weights_dir']
    # tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_dict[model_name]['tokenizer'])
    inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = inputs.input_ids
    # 输入序列
    inputs = token_ids[:, :-1]  
    # 目标序列
    targets = token_ids[:, 1:]  

    model = LlamaForCausalLM.from_pretrained(model_dict[model_name]['hf_model'])
    # hf 前向
    outputs = model(input_ids=inputs)
    hf_logits = outputs.logits

    # naked_llama 前向
    llama2 = LlamaModel(config)
    manual_logits = llama2.forward(inputs) 

    # grad_output = 0.1 * torch.randn_like(logits)
    # hf_logits.backward(grad_output, retain_graph=True)
    
    # 交叉熵计算loss
    official_loss = F.cross_entropy(hf_logits.view(-1, hf_logits.size(-1)), targets.view(-1), reduction='mean')
    official_loss.backward()

    # 保存hf_model自动求导梯度
    hf_grads = {}
    
    hf_grads['lm_head_weight_grad'] = model.lm_head.weight.grad.clone()
    hf_grads['final_norm_weight_grad'] = model.model.norm.weight.grad.clone()
    
    for i, block in enumerate(model.model.layers):
        hf_grads[f'block_{i}_input_layernorm_weight_grad'] = block.input_layernorm.weight.grad.clone()
        hf_grads[f'block_{i}_post_attention_layernorm_weight_grad'] = block.post_attention_layernorm.weight.grad.clone()
        hf_grads[f'block_{i}_mlp_up_proj_weight_grad'] = block.mlp.up_proj.weight.grad.clone()
        hf_grads[f'block_{i}_mlp_gate_proj_weight_grad'] = block.mlp.gate_proj.weight.grad.clone()
        hf_grads[f'block_{i}_mlp_down_proj_weight_grad'] = block.mlp.down_proj.weight.grad.clone()
        hf_grads[f'block_{i}_self_attn_q_proj_weight_grad'] = block.self_attn.q_proj.weight.grad.clone()
        hf_grads[f'block_{i}_self_attn_k_proj_weight_grad'] = block.self_attn.k_proj.weight.grad.clone()
        hf_grads[f'block_{i}_self_attn_v_proj_weight_grad'] = block.self_attn.v_proj.weight.grad.clone()
        hf_grads[f'block_{i}_self_attn_o_proj_weight_grad'] = block.self_attn.o_proj.weight.grad.clone()
        if block.self_attn.q_proj.bias is not None and hasattr(block.self_attn.q_proj.bias, 'grad'):
            hf_grads[f'block_{i}_self_attn_q_proj_bias_grad'] = block.self_attn.q_proj.bias.grad.clone()
        if block.self_attn.k_proj.bias is not None and hasattr(block.self_attn.k_proj.bias, 'grad'):
            hf_grads[f'block_{i}_self_attn_k_proj_bias_grad'] = block.self_attn.k_proj.bias.grad.clone()
        if block.self_attn.v_proj.bias is not None and hasattr(block.self_attn.v_proj.bias, 'grad'):
            hf_grads[f'block_{i}_self_attn_v_proj_bias_grad'] = block.self_attn.v_proj.bias.grad.clone()
        if block.self_attn.o_proj.bias is not None and hasattr(block.self_attn.o_proj.bias, 'grad'):
            hf_grads[f'block_{i}_self_attn_o_proj_bias_grad'] = block.self_attn.o_proj.bias.grad.clone()
    
    hf_grads['embedding_weight_grad'] = model.model.embed_tokens.weight.grad.clone()

    cross_entropy_manual = CrossEntropy(reduction='mean')
    manual_loss = cross_entropy_manual.forward(manual_logits.view(-1, manual_logits.size(-1)), targets.view(-1))
    grad_output = cross_entropy_manual.backward(targets.view(-1), manual_loss)
    manual_grads = llama2.backward(grad_output)

    for key in hf_grads:
        hf_grad = hf_grads[key]
        hw_grad = manual_grads.get(key)
        if hw_grad is None:
            print(f'Missing grad for {key} in manual backward.')
            continue
        try:
            torch.testing.assert_close(hf_grad, hw_grad)

            print(f'{key}: grads match.')
        except AssertionError as e:
            print(f'{key}: grads do not match. {e}')

if __name__ == '__main__':

    test_llama_backward_manual_class()
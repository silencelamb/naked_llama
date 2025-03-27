# DeepSeek DP attn优化分析

TP_size = 16
EP_size = 4

# hardware parameters
GB = 1024 * 1024 * 1024
us = 1e-6 # 1us
C2C_latency = 8 * us
C2C_BW = 50 * 0.4 * GB
DDR_BW = 120 * GB

# Model parameters
hidden_dim = 7168
head_num = 128
head_dim_v = 128
rope_dim = 64
head_dim_qk = head_dim_v + rope_dim
lora_rank_k = 512
lora_rank_q = 1536
num_layers = 61


def benifit(kv_len, TP_size):
    # 应该是只在decode有收益
    # prefill时，用native版本，本身是MHA，DP-attn不会提高KV的数据复用
    # 如果prefill时，用absorb版本呢？也许也挺好？
    kv_cache = kv_len * (lora_rank_k + rope_dim)
    kv_ddr_time = kv_cache * 2 /DDR_BW * (1-1/TP_size)  # 每个DDR都读取完整的kv_cache -> 每个DDR只读取1/TP_size的kv_cache
    return kv_ddr_time * num_layers

# SGLang version
pure_dp_dp_attn = {
    "name": "Pure_DP_DP_Attn"
}
W_UQ = lora_rank_q * head_num * head_dim_v
W_QR = lora_rank_q * head_num * rope_dim
W_UK = lora_rank_k * head_num * head_dim_v
W_UV = lora_rank_k * head_num * head_dim_v
W_O = head_num * head_dim_v * hidden_dim

total_weight = W_UQ + W_QR + W_UK + W_UV + W_O

def pure_dp_overhead(q_len, TP_size):
    ddr_time =  total_weight * 2 * (TP_size - 1) / TP_size / DDR_BW
    o_activation_num  = q_len * hidden_dim
    # 少了一个all-reduce
    # 但是多了1个all-gather，或者说多了all-to-all到专家之后的broad-cast
    # 假设两者抵消
    all_reduce_time = 2 * (TP_size - 1) * C2C_latency + 2 * o_activation_num / C2C_BW * 2
    # overhead_time = ddr_time - all_reduce_time
    overhead_time = ddr_time
    return overhead_time * num_layers

pure_dp_dp_attn["overhead_func"] = lambda q_len, TP_size: pure_dp_overhead(q_len, TP_size)
pure_dp_dp_attn["benifit_func"] = lambda kv_len, TP_size: benifit(kv_len, TP_size)


# Google version
tp_with_dp_attn = {
    "name": "TP_With_DP_Attn"
}

def tp_dp_overhead(q_len, TP_size, stage='prefill'):
    if stage == 'prefill':
        q_comm = q_len * head_num * head_dim_qk
        k_comm = q_len * head_num * head_dim_v   # rope 64 is replicated
        v_comm = q_len * head_num * head_dim_v
        total_comm = q_comm + k_comm + v_comm
    else:
        q_len = 1
        q_comm = q_len * head_num * (lora_rank_k + rope_dim)
        total_comm = q_comm
    
    comm_time = total_comm * 2 / C2C_BW + (TP_size - 1) * 2 * C2C_latency
    comm_time = comm_time * 2 # 2次 all-to-all
    return comm_time * num_layers

tp_with_dp_attn["overhead_func"] = lambda q_len, TP_size, stage: tp_dp_overhead(q_len, TP_size, stage)
tp_with_dp_attn["benifit_func"] = lambda kv_len, TP_size: benifit(kv_len, TP_size)

import matplotlib.pyplot as plt
import numpy as np

# 计算不同q_len下的overhead和benefit
q_lens = np.logspace(0, 15, num=18, base=2).astype(int)  # 从1到32768的2的幂
kv_lens = q_lens.copy()

# 计算Pure DP方法的开销和收益
pure_dp_overheads = [pure_dp_dp_attn["overhead_func"](q_len, TP_size) for q_len in q_lens]
pure_dp_benefits = [pure_dp_dp_attn["benifit_func"](kv_len, TP_size) for kv_len in kv_lens]
pure_dp_net_benefits = [b - o for b, o in zip(pure_dp_benefits, pure_dp_overheads)]

# 计算TP with DP方法的开销和收益
tp_dp_overheads_prefill = [tp_with_dp_attn["overhead_func"](q_len, TP_size, 'prefill') for q_len in q_lens]
tp_dp_overheads_decode = [tp_with_dp_attn["overhead_func"](q_len, TP_size, 'decode') for q_len in q_lens]

tp_dp_benefits = [tp_with_dp_attn["benifit_func"](kv_len, TP_size) for kv_len in kv_lens]
tp_dp_net_benefits_prefill = [b - o for b, o in zip(tp_dp_benefits, tp_dp_overheads_prefill)]
tp_dp_net_benefits_decode = [b - o for b, o in zip(tp_dp_benefits, tp_dp_overheads_decode)]

print(f"Pure DP Overheads : {pure_dp_overheads[0]/us} us")
print(f"TP DP Overheads (Decode) : {tp_dp_overheads_decode[-1]/us} us")
print(f"TP DP Benefits: {tp_dp_benefits[-1]/us} us")

# 创建绘图数据
plt.figure(figsize=(15, 10))

# 1. 绘制Overhead对比
plt.subplot(2, 2, 1)
plt.loglog(q_lens, [o*1e6 for o in pure_dp_overheads], 'o-', label='Pure DP')
plt.loglog(q_lens, [o*1e6 for o in tp_dp_overheads_prefill], 's-', label='TP+DP (Prefill)')
plt.loglog(q_lens, [o*1e6 for o in tp_dp_overheads_decode], '^-', label='TP+DP (Decode)')
plt.xlabel('Query Length')
plt.ylabel('Overhead (μs)')
plt.title('Overhead Comparison')
plt.grid(True, which="both", ls="--")
plt.legend()

# 2. 绘制Benefit对比
plt.subplot(2, 2, 2)
plt.loglog(kv_lens, [b*1e6 for b in pure_dp_benefits], 'o-', label='Pure DP')
plt.loglog(kv_lens, [b*1e6 for b in tp_dp_benefits], 's-', label='TP+DP')
plt.xlabel('KV Cache Length')
plt.ylabel('Benefit (μs)')
plt.title('Benefit Comparison')
plt.grid(True, which="both", ls="--")
plt.legend()

# 3. 绘制Net Benefit对比
plt.subplot(2, 2, 3)
plt.semilogx(q_lens, [n*1e6 for n in pure_dp_net_benefits], 'o-', label='Pure DP')
plt.semilogx(q_lens, [n*1e6 for n in tp_dp_net_benefits_prefill], 's-', label='TP+DP (Prefill)')
plt.semilogx(q_lens, [n*1e6 for n in tp_dp_net_benefits_decode], '^-', label='TP+DP (Decode)')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Sequence Length')
plt.ylabel('Net Benefit (μs)')
plt.title('Net Benefit Comparison')
plt.grid(True, which="both", ls="--")
plt.legend()

# 4. 绘制Benefit/Overhead比例
plt.subplot(2, 2, 4)
plt.semilogx(q_lens, [b/max(1e-10, o) for b, o in zip(pure_dp_benefits, pure_dp_overheads)], 'o-', label='Pure DP')
plt.semilogx(q_lens, [b/max(1e-10, o) for b, o in zip(tp_dp_benefits, tp_dp_overheads_prefill)], 's-', label='TP+DP (Prefill)')
plt.semilogx(q_lens, [b/max(1e-10, o) for b, o in zip(tp_dp_benefits, tp_dp_overheads_decode)], '^-', label='TP+DP (Decode)')
plt.axhline(y=1, color='r', linestyle='-')
plt.xlabel('Sequence Length')
plt.ylabel('Benefit/Overhead Ratio')
plt.title('Efficiency Ratio (>1 means beneficial)')
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()

# 显示图表
plt.show()
# DeepSeek DP attn优化分析

TP_size = 16
EP_size = 4
Batch_size = TP_size

# hardware parameters
GB = 1024 * 1024 * 1024
us = 1e-6 # 1us
C2C_latency = 8 * us
C2C_BW = 50 * 0.4 * GB
DDR_BW = 120 * GB
HBM_BW = 3.35 * 0.9 * 1024 * GB

Parallel_COMM_TimeStep = {
    8: 3,
    16: 4,
    32: 5
}

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
    kv_cache = kv_len * (lora_rank_k + rope_dim) * num_layers * Batch_size
    kv_ddr_time = kv_cache * 2 /DDR_BW * (1-1/TP_size)  # 每个DDR都读取完整的kv_cache -> 每个DDR只读取1/TP_size的kv_cache
    return kv_ddr_time

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

def pure_dp_overhead(q_len, TP_size, bw=DDR_BW):
    ddr_time = total_weight * 2 * (TP_size - 1) / TP_size / bw
    o_activation_num = q_len * hidden_dim * Batch_size
    # 少了一个all-reduce
    # 但是多了1个all-gather，或者说多了all-to-all到专家之后的broad-cast
    # 假设两者抵消
    all_reduce_time = 2 * (TP_size - 1) * C2C_latency + 2 * o_activation_num / C2C_BW * 2
    # overhead_time = ddr_time - all_reduce_time
    overhead_time = ddr_time
    return overhead_time * num_layers

pure_dp_dp_attn["overhead_func"] = lambda q_len, TP_size, bw: pure_dp_overhead(q_len, TP_size, bw)
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
        # 在decode模式下，我们通常关心单token生成的情况
        q_comm = q_len * head_num * (lora_rank_k + rope_dim)
        total_comm = q_comm
    # 一般 Batch_size/TP_size = 1
    total_comm = total_comm * Batch_size/TP_size
    ring_comm_time = total_comm * 2 / C2C_BW + (TP_size - 1)  * C2C_latency
    parallel_comm_time = (total_comm * 2 / C2C_BW + C2C_latency) * Parallel_COMM_TimeStep[TP_size]
    comm_time = min(ring_comm_time, parallel_comm_time)
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
pure_dp_overheads = [pure_dp_dp_attn["overhead_func"](q_len, TP_size, DDR_BW) for q_len in q_lens]
pure_dp_overheads_H100 = [pure_dp_dp_attn["overhead_func"](q_len, TP_size, HBM_BW) for q_len in q_lens]
pure_dp_benefits = [pure_dp_dp_attn["benifit_func"](kv_len, TP_size) for kv_len in kv_lens]

# 计算TP with DP方法的开销和收益
tp_dp_overheads_prefill = [tp_with_dp_attn["overhead_func"](q_len, TP_size, 'prefill') for q_len in q_lens]
# 对于decode，我们使用q_len=1，因为这是典型的自回归生成场景
tp_dp_overheads_decode = [tp_with_dp_attn["overhead_func"](1, TP_size, 'decode') for _ in q_lens]

tp_dp_benefits = [tp_with_dp_attn["benifit_func"](kv_len, TP_size) for kv_len in kv_lens]

print(f"Pure DP Overheads : {pure_dp_overheads[0]/us} us")
print(f"Pure DP Overheads H100 : {pure_dp_overheads_H100[0]/us} us")
print(f"TP DP Overheads (Decode) : {tp_dp_overheads_decode[-1]/us} us")
print(f"TP DP Benefits: {tp_dp_benefits[-1]/us} us")

# 创建绘图数据
plt.figure(figsize=(20, 10))  # 增加高度以适应额外的图表

# 1. 绘制Overhead对比
plt.subplot(1, 2, 1)
plt.loglog(q_lens, [o*1e6 for o in pure_dp_overheads], 'o-', label='Pure DP')
plt.loglog(q_lens, [o*1e6 for o in tp_dp_overheads_prefill], 's-', label='TP+DP (Prefill)')
plt.loglog(q_lens, [o*1e6 for o in tp_dp_overheads_decode], '^-', label='TP+DP (Decode)')
plt.xlabel('Query Length')
plt.ylabel('Overhead (μs)')
plt.title('Overhead Comparison')
plt.grid(True, which="both", ls="--")
plt.legend()


# 2. 新增图表：Pure DP Overheads、TP DP Overheads (Decode)和TP DP Benefits比较
plt.subplot(1, 2, 2)
plt.loglog(q_lens, [o*1e6 for o in pure_dp_overheads], 'o-', label='Pure DP Overheads')
plt.loglog(q_lens, [o*1e6 for o in pure_dp_overheads_H100], 'o-', label='Pure DP Overheads_H100')
plt.loglog(q_lens, [o*1e6 for o in tp_dp_overheads_decode], '^-', label='TP+DP Overheads (Decode)')
plt.loglog(kv_lens, [b*1e6 for b in tp_dp_benefits], 's-', label='TP+DP Benefits')
plt.xlabel('Sequence Length')
plt.ylabel('Time (μs)')
plt.title('Comparison of Overheads and Benefits')
plt.grid(True, which="both", ls="--")
plt.legend()

plt.tight_layout()

# 显示图表
plt.show()
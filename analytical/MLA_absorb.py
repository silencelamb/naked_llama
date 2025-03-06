import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import base64

# Model parameters
hidden_dim = 7168
head_num = 128
head_dim_v = 128
rope_dim = 64
head_dim_qk = head_dim_v + rope_dim
lora_rank_k = 512
lora_rank_q = 1536

# vanilla version
def num_float_ops_vanilla(q_len, kv_len):
    return (q_len * lora_rank_q * head_num * (head_dim_v + rope_dim) +  # from c_Q to q_pe and q_nope, corresponding to q_b_proj
            kv_len * head_num * lora_rank_k * (head_dim_v + head_dim_v) +  # from compressed_kv to k_nop and value_states, corresponding to kv_b_proj
            head_num * (q_len * head_dim_qk * kv_len + q_len * kv_len * head_dim_v) +  # 128 heads MHA
            q_len * (head_num * head_dim_v) * hidden_dim)  # from MHA output to output hidden states, corresponding to o_proj

def mem_footprint_vanilla(q_len, kv_len):
    return (q_len * lora_rank_q + lora_rank_q * (head_num * head_dim_qk) +  # q_lora, q_up_weight
            kv_len * (lora_rank_k + rope_dim) + lora_rank_k * (head_num * (head_dim_v + head_dim_v)) +  # cached_k_lora, W_UK, W_UV
            head_num * (q_len * head_dim_qk + head_dim_qk * kv_len + head_dim_v * kv_len) +  # 128 heads MHA, q, k, v
            q_len * (head_num * head_dim_v) + (head_num * head_dim_v) * hidden_dim)  # attn_output, o_proj weight

# absorbed version weight multiply
def num_float_ops_mat_absorb_mul(q_len, kv_len):
    return (q_len * lora_rank_q * head_num * (head_dim_v + rope_dim) +  # from c_Q to q_pe and q_nope, corresponding to q_b_proj
            q_len * head_num * head_dim_v * lora_rank_k +  # from q_nope to q_nope 512 dim, corresponding to W_UK
            head_num * (q_len * (lora_rank_k + rope_dim) * kv_len + q_len * kv_len * lora_rank_k) +  # 128 heads MQA
            q_len * head_num * lora_rank_k * head_dim_v +  # MHA output 512 dim => 128 dim, corresponding to W_UV_O
            q_len * head_num * head_dim_v * hidden_dim)  # from MHA output to output hidden states, corresponding to o_proj

def mem_footprint_mat_absorb_mul(q_len, kv_len):
    return (q_len * lora_rank_q + lora_rank_q * (head_num * head_dim_qk) +  # q_lora, q_up_weight
            q_len * (head_num * head_dim_qk) +  # q dim 192
            kv_len * (lora_rank_k + rope_dim) + lora_rank_k * (head_num * (head_dim_v + head_dim_v)) +  # cached_k_lora, W_UK, W_UV
            head_num * (q_len * (lora_rank_k + rope_dim)) +  # 128 heads Q
            q_len * head_num * lora_rank_k +  # atten output 512 dim
            q_len * (head_num * head_dim_v) + (head_num * head_dim_v) * hidden_dim)  # attn_output, o_proj weight

# absorbed version full absorb
def num_float_ops_mat_absorb_all(q_len, kv_len):
    return (q_len * lora_rank_q * head_num *  rope_dim +  # from c_Q to q_pe, corresponding to q_b_proj
            q_len * head_num * lora_rank_q * lora_rank_k +  # from c_Q to q_nope, corresponding to W_UQUK
            head_num * (q_len * (lora_rank_k + rope_dim) * kv_len + q_len * kv_len * lora_rank_k) +  # 128 heads MQA
            q_len * head_num * lora_rank_k * hidden_dim)  # from MHA output to output hidden states, corresponding to W_UV_O

def mem_footprint_mat_absorb_all(q_len, kv_len):
    return (q_len * lora_rank_q + lora_rank_q * head_num * rope_dim +  # q_lora, q_rope_weight
            q_len * (head_num * rope_dim) +  # qrope
            head_num * lora_rank_q * lora_rank_k +  # W_UQUK
            kv_len * (lora_rank_k + rope_dim) +  # cached_k_lora
            head_num * q_len * (lora_rank_k + rope_dim) +  # 128 heads Q
            q_len * (head_num * lora_rank_k) +  # attn output
            (head_num * lora_rank_k) * hidden_dim)  # W_UV_O

# Test with different kv_len values
kv_lens = [8, 32, 128, 512, 1024, 2048, 4096, 10240, 20480, 64000, 102400]

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Create dataframes to store results
prefill_ops_df = pd.DataFrame(columns=["kv_len", "Vanilla", "Weight Multiply", "Full Absorb", "W.M. / Vanilla", "F.A. / Vanilla"])
prefill_mem_df = pd.DataFrame(columns=["kv_len", "Vanilla", "Weight Multiply", "Full Absorb", "W.M. / Vanilla", "F.A. / Vanilla"])
decode_ops_df = pd.DataFrame(columns=["kv_len", "Vanilla", "Weight Multiply", "Full Absorb", "W.M. / Vanilla", "F.A. / Vanilla"])
decode_mem_df = pd.DataFrame(columns=["kv_len", "Vanilla", "Weight Multiply", "Full Absorb", "W.M. / Vanilla", "F.A. / Vanilla"])

# Lists to store normalized results for plotting
plot_kv_lens = []
prefill_ops_wm_norm = []
prefill_ops_fa_norm = []
prefill_mem_wm_norm = []
prefill_mem_fa_norm = []
decode_ops_wm_norm = []
decode_ops_fa_norm = []
decode_mem_wm_norm = []
decode_mem_fa_norm = []

# Calculate and store results
for kv_len in kv_lens:
    # Prefill phase (q_len = kv_len)
    prefill_vanilla_ops = num_float_ops_vanilla(kv_len, kv_len)
    prefill_wm_ops = num_float_ops_mat_absorb_mul(kv_len, kv_len)
    prefill_fa_ops = num_float_ops_mat_absorb_all(kv_len, kv_len)
    prefill_wm_ops_ratio = prefill_wm_ops / prefill_vanilla_ops
    prefill_fa_ops_ratio = prefill_fa_ops / prefill_vanilla_ops
    
    prefill_vanilla_mem = mem_footprint_vanilla(kv_len, kv_len)
    prefill_wm_mem = mem_footprint_mat_absorb_mul(kv_len, kv_len)
    prefill_fa_mem = mem_footprint_mat_absorb_all(kv_len, kv_len)
    prefill_wm_mem_ratio = prefill_wm_mem / prefill_vanilla_mem
    prefill_fa_mem_ratio = prefill_fa_mem / prefill_vanilla_mem
    
    # Decode phase (q_len = 1)
    q_len = 1
    decode_vanilla_ops = num_float_ops_vanilla(q_len, kv_len)
    decode_wm_ops = num_float_ops_mat_absorb_mul(q_len, kv_len)
    decode_fa_ops = num_float_ops_mat_absorb_all(q_len, kv_len)
    decode_wm_ops_ratio = decode_wm_ops / decode_vanilla_ops
    decode_fa_ops_ratio = decode_fa_ops / decode_vanilla_ops
    
    decode_vanilla_mem = mem_footprint_vanilla(q_len, kv_len)
    decode_wm_mem = mem_footprint_mat_absorb_mul(q_len, kv_len)
    decode_fa_mem = mem_footprint_mat_absorb_all(q_len, kv_len)
    decode_wm_mem_ratio = decode_wm_mem / decode_vanilla_mem
    decode_fa_mem_ratio = decode_fa_mem / decode_vanilla_mem
    
    # Add to dataframes
    prefill_ops_df = pd.concat([prefill_ops_df, pd.DataFrame({
        "kv_len": [kv_len],
        "Vanilla": [prefill_vanilla_ops],
        "Weight Multiply": [prefill_wm_ops],
        "Full Absorb": [prefill_fa_ops],
        "W.M. / Vanilla": [prefill_wm_ops_ratio],
        "F.A. / Vanilla": [prefill_fa_ops_ratio]
    })], ignore_index=True)
    
    prefill_mem_df = pd.concat([prefill_mem_df, pd.DataFrame({
        "kv_len": [kv_len],
        "Vanilla": [prefill_vanilla_mem],
        "Weight Multiply": [prefill_wm_mem],
        "Full Absorb": [prefill_fa_mem],
        "W.M. / Vanilla": [prefill_wm_mem_ratio],
        "F.A. / Vanilla": [prefill_fa_mem_ratio]
    })], ignore_index=True)
    
    decode_ops_df = pd.concat([decode_ops_df, pd.DataFrame({
        "kv_len": [kv_len],
        "Vanilla": [decode_vanilla_ops],
        "Weight Multiply": [decode_wm_ops],
        "Full Absorb": [decode_fa_ops],
        "W.M. / Vanilla": [decode_wm_ops_ratio],
        "F.A. / Vanilla": [decode_fa_ops_ratio]
    })], ignore_index=True)
    
    decode_mem_df = pd.concat([decode_mem_df, pd.DataFrame({
        "kv_len": [kv_len],
        "Vanilla": [decode_vanilla_mem],
        "Weight Multiply": [decode_wm_mem],
        "Full Absorb": [decode_fa_mem],
        "W.M. / Vanilla": [decode_wm_mem_ratio],
        "F.A. / Vanilla": [decode_fa_mem_ratio]
    })], ignore_index=True)
    
    # Store for plotting
    plot_kv_lens.append(kv_len)
    prefill_ops_wm_norm.append(prefill_wm_ops_ratio)
    prefill_ops_fa_norm.append(prefill_fa_ops_ratio)
    prefill_mem_wm_norm.append(prefill_wm_mem_ratio)
    prefill_mem_fa_norm.append(prefill_fa_mem_ratio)
    decode_ops_wm_norm.append(decode_wm_ops_ratio)
    decode_ops_fa_norm.append(decode_fa_ops_ratio)
    decode_mem_wm_norm.append(decode_wm_mem_ratio)
    decode_mem_fa_norm.append(decode_fa_mem_ratio)

# Save as CSV
prefill_ops_df.to_csv('results/prefill_ops.csv', index=False)
prefill_mem_df.to_csv('results/prefill_mem.csv', index=False)
decode_ops_df.to_csv('results/decode_ops.csv', index=False)
decode_mem_df.to_csv('results/decode_mem.csv', index=False)

# Function to convert matplotlib figure to base64 encoded image for HTML display
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

# Create bar plots
fig1 = plt.figure(figsize=(20, 16))

# Convert kv_lens to strings for better display
kv_lens_str = [str(x) for x in plot_kv_lens]

# Plot 1: Prefill Phase - Computational Cost
plt.subplot(2, 2, 1)
x = np.arange(len(kv_lens_str))
width = 0.3
plt.bar(x - width, [1] * len(kv_lens_str), width, label='Vanilla')
plt.bar(x, prefill_ops_wm_norm, width, label='Weight Multiply')
plt.bar(x + width, prefill_ops_fa_norm, width, label='Full Absorb')
plt.xlabel('KV Length')
plt.ylabel('Normalized FLOPs (Vanilla = 1)')
plt.title('Prefill Phase - Computational Cost')
plt.xticks(x, kv_lens_str, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Prefill Phase - Memory Footprint
plt.subplot(2, 2, 2)
plt.bar(x - width, [1] * len(kv_lens_str), width, label='Vanilla')
plt.bar(x, prefill_mem_wm_norm, width, label='Weight Multiply')
plt.bar(x + width, prefill_mem_fa_norm, width, label='Full Absorb')
plt.xlabel('KV Length')
plt.ylabel('Normalized Memory (Vanilla = 1)')
plt.title('Prefill Phase - Memory Footprint')
plt.xticks(x, kv_lens_str, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 3: Decode Phase - Computational Cost
plt.subplot(2, 2, 3)
plt.bar(x - width, [1] * len(kv_lens_str), width, label='Vanilla')
plt.bar(x, decode_ops_wm_norm, width, label='Weight Multiply')
plt.bar(x + width, decode_ops_fa_norm, width, label='Full Absorb')
plt.xlabel('KV Length')
plt.ylabel('Normalized FLOPs (Vanilla = 1)')
plt.title('Decode Phase - Computational Cost')
plt.xticks(x, kv_lens_str, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Plot 4: Decode Phase - Memory Footprint
plt.subplot(2, 2, 4)
plt.bar(x - width, [1] * len(kv_lens_str), width, label='Vanilla')
plt.bar(x, decode_mem_wm_norm, width, label='Weight Multiply')
plt.bar(x + width, decode_mem_fa_norm, width, label='Full Absorb')
plt.xlabel('KV Length')
plt.ylabel('Normalized Memory (Vanilla = 1)')
plt.title('Decode Phase - Memory Footprint')
plt.xticks(x, kv_lens_str, rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('results/mla_comparison.png', dpi=300)
bar_img_str = fig_to_base64(fig1)
plt.close(fig1)

# Additional line plots for ratio trends
fig2 = plt.figure(figsize=(20, 16))

# Plot 1: Prefill Phase - Computational Cost Trends
plt.subplot(2, 2, 1)
plt.plot(plot_kv_lens, [1] * len(plot_kv_lens), 'k-', label='Vanilla')
plt.plot(plot_kv_lens, prefill_ops_wm_norm, 'b-o', label='Weight Multiply')
plt.plot(plot_kv_lens, prefill_ops_fa_norm, 'r-^', label='Full Absorb')
plt.xlabel('KV Length (log scale)')
plt.ylabel('Normalized FLOPs (Vanilla = 1)')
plt.title('Prefill Phase - Computational Cost Trends')
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plot 2: Prefill Phase - Memory Footprint Trends
plt.subplot(2, 2, 2)
plt.plot(plot_kv_lens, [1] * len(plot_kv_lens), 'k-', label='Vanilla')
plt.plot(plot_kv_lens, prefill_mem_wm_norm, 'b-o', label='Weight Multiply')
plt.plot(plot_kv_lens, prefill_mem_fa_norm, 'r-^', label='Full Absorb')
plt.xlabel('KV Length (log scale)')
plt.ylabel('Normalized Memory (Vanilla = 1)')
plt.title('Prefill Phase - Memory Footprint Trends')
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plot 3: Decode Phase - Computational Cost Trends
plt.subplot(2, 2, 3)
plt.plot(plot_kv_lens, [1] * len(plot_kv_lens), 'k-', label='Vanilla')
plt.plot(plot_kv_lens, decode_ops_wm_norm, 'b-o', label='Weight Multiply')
plt.plot(plot_kv_lens, decode_ops_fa_norm, 'r-^', label='Full Absorb')
plt.xlabel('KV Length (log scale)')
plt.ylabel('Normalized FLOPs (Vanilla = 1)')
plt.title('Decode Phase - Computational Cost Trends')
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Plot 4: Decode Phase - Memory Footprint Trends
plt.subplot(2, 2, 4)
plt.plot(plot_kv_lens, [1] * len(plot_kv_lens), 'k-', label='Vanilla')
plt.plot(plot_kv_lens, decode_mem_wm_norm, 'b-o', label='Weight Multiply')
plt.plot(plot_kv_lens, decode_mem_fa_norm, 'r-^', label='Full Absorb')
plt.xlabel('KV Length (log scale)')
plt.ylabel('Normalized Memory (Vanilla = 1)')
plt.title('Decode Phase - Memory Footprint Trends')
plt.xscale('log')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.tight_layout()
plt.savefig('results/mla_trend_comparison.png', dpi=300)
trend_img_str = fig_to_base64(fig2)
plt.close(fig2)

# Generate HTML with tables and embedded images
html_output = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MLA Implementation Comparison</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 30px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: right;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
            text-align: center;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f1f1f1;
        }}
        .img-container {{
            text-align: center;
            margin: 20px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        .section {{
            margin-bottom: 40px;
        }}
    </style>
</head>
<body>
    <h1>MLA Implementation Comparison Analysis</h1>
    
    <div class="section">
        <h2>Prefill Phase - Computational Cost (FLOPs)</h2>
        {prefill_ops_df.to_html(index=False, float_format=lambda x: '{:.2e}'.format(x) if isinstance(x, float) and x > 1000 else '{:.5f}'.format(x))}
    </div>
    
    <div class="section">
        <h2>Prefill Phase - Memory Footprint</h2>
        {prefill_mem_df.to_html(index=False, float_format=lambda x: '{:.2e}'.format(x) if isinstance(x, float) and x > 1000 else '{:.5f}'.format(x))}
    </div>
    
    <div class="section">
        <h2>Decode Phase - Computational Cost (FLOPs)</h2>
        {decode_ops_df.to_html(index=False, float_format=lambda x: '{:.2e}'.format(x) if isinstance(x, float) and x > 1000 else '{:.5f}'.format(x))}
    </div>
    
    <div class="section">
        <h2>Decode Phase - Memory Footprint</h2>
        {decode_mem_df.to_html(index=False, float_format=lambda x: '{:.2e}'.format(x) if isinstance(x, float) and x > 1000 else '{:.5f}'.format(x))}
    </div>
    
    <div class="section">
        <h2>Bar Charts - Normalized Comparison (Vanilla = 1)</h2>
        <div class="img-container">
            <img src="data:image/png;base64,{bar_img_str}" alt="Bar Charts Comparison">
        </div>
    </div>
    
    <div class="section">
        <h2>Trend Charts - Normalized Comparison Across KV Lengths</h2>
        <div class="img-container">
            <img src="data:image/png;base64,{trend_img_str}" alt="Trend Charts">
        </div>
    </div>
</body>
</html>
"""

# Save HTML to file
with open('results/mla_comparison_report.html', 'w') as f:
    f.write(html_output)

print("\nResults saved in the 'results' directory:")
print(" - CSV files with detailed metrics")
print(" - PNG images of bar charts and trend lines")
print(" - HTML report with all tables and charts (results/mla_comparison_report.html)")
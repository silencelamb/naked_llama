import torch
torch.set_printoptions(linewidth=200) 

# 固定数字（除数）
divisor = 3.1416

# 随机生成范围在 2048 ~ 4096 的数值
data_fp32 = torch.rand(10).cuda() * 2048 + 2048

# 方式1: 使用 bf16 直接除以固定数字
data_bf16 = data_fp32.to(torch.bfloat16)
result_bf16_direct = data_bf16 / divisor

# 方式2: 使用 fp32 进行除法，然后转换为 bf16
result_fp32 = data_fp32 / divisor
result_bf16_after = result_fp32.to(torch.bfloat16)

# 比较两种方式的误差
abs_diff = (result_bf16_direct - result_bf16_after).abs()

# 输出结果
print("原始 FP32 数据: ", data_fp32)
print("bf16 除法后结果: ", result_bf16_direct)
print("fp32 除法后的结果: ", result_fp32)
print("fp32 除法后转换为 bf16 的结果: ", result_bf16_after)
print("两者之间的绝对误差: ", abs_diff)

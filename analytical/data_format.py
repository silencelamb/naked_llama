import numpy as np

# 打印表头
print(f"{'Min':<30}{'Max':<15}{'Interval':<15}")

# bfloat16 的指数部分范围是 -126 到 127
bias = 127
mantissa_bits = 7
exponent_bits = 8
# 非规范数
real_exponent = 1-bias
min_val = 0
max_val = 2** real_exponent
interval = 2**(real_exponent - mantissa_bits)
print(f"0\t{min_val:<15.4e}\t2**{real_exponent}\t{max_val:<15.4e}\t2**{(real_exponent - mantissa_bits)}\t{interval:.4e}")
# 规范数
for exponent in range(1, 200):
    real_exponent = exponent - bias
    min_val = 2**real_exponent
    max_val = 2**(real_exponent + 1)
    interval = 2**(real_exponent - mantissa_bits)  # bfloat16 的尾数部分有 7 位
    print(f"2**{real_exponent}\t{min_val:<15.4e}\t2**{real_exponent+1}\t{max_val:<15.4e}\t2**{(real_exponent - mantissa_bits)}\t{interval:.4e}")


# 计算并打印 float16 的范围和间隔
# 打印表头
print(f"{'Min':<30}{'Max':<30}{'Interval':<15}")

bias = 15
exponent_bits = 5
mantissa_bits = 10

# 计算并打印 float16 的范围和间隔
# 非规范数
real_exponent = 1-bias
min_val = 0
max_val = 2** real_exponent
interval = 2**(real_exponent - mantissa_bits)
print(f"0\t{min_val:<15.4e}\t2**{real_exponent}\t{max_val:<15.4e}\t2**{(real_exponent - mantissa_bits)}\t{interval:.4e}")

# 规范数
for exponent in range(1, 15):
    real_exponent = exponent - bias
    min_val = 2**real_exponent
    max_val = 2**(real_exponent + 1)
    interval = 2**(real_exponent - mantissa_bits)  # float16 的尾数部分有 10 位
    print(f"2**{real_exponent}\t{min_val:<15.4e}\t2**{real_exponent+1}\t{max_val:<15.4e}\t2**{(real_exponent - mantissa_bits)}\t{interval:.4e}")

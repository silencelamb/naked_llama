from .hw_params import GB, C2C

'''
This file contains the analytical model for communication time.
Reference:
1. Google, MLSys2023, Efficiently Scaling Transformer Inference, Page 16
2. Oneflow, 手把手推导Ring All-reduce的数学性质, https://mp.weixin.qq.com/s/0D1UESC4vO7cqNZnnN0_vQ

'''

def reduce_scatter(C2C, bytes, device_num):
    comm_time = bytes/device_num/C2C * (device_num-1)
    return comm_time

def all_gather(C2C, bytes, device_num):
    comm_time = bytes/device_num/C2C * (device_num-1)
    return comm_time

def all_reduce(C2C, bytes, device_num):
    comm_time = bytes/device_num/C2C * (device_num-1) * 2
    return comm_time
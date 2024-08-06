import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    # os.environ['MASTER_ADDR'] = 'localhost'
    # docker 内，若是bridge 网络，使用容器的 ip，而不是 localhost
    # 使用localhost，会报错： (errno: 101 - Network is unreachable)
    # os.environ['MASTER_ADDR'] = '172.17.0.4'
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def example_all_reduce(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    tensor = torch.ones(1, device=device) * rank

    # 同步所有进程，确保计时准确
    # warm-up
    dist.barrier()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    # 真正执行
    dist.barrier()
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f'Rank {rank} has tensor {tensor} after all_reduce')
    print(f'Rank {rank} all_reduce took {elapsed_time*1000} ms')
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(example_all_reduce, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    # python nccl_all_reduce_mp.py
    main()


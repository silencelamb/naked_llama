import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

def setup(rank, world_size):
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
    torch.cuda.synchronize(device)
    
    # 真正执行
    dist.barrier()
    start_time = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f'Rank {rank} has tensor {tensor} after all_reduce')
    print(f'Rank {rank} all_reduce took {elapsed_time*1000} ms')
    
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    rank = int(os.environ['LOCAL_RANK'])
    example_all_reduce(rank, world_size)

if __name__ == "__main__":
    # torchrun --nproc_per_node=8 nccl_all_reduce_torchrun.py
    # nvlink all reduce  A100 8卡， 0.06 ms， 60us；
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 nccl_all_reduce_torchrun.py
    # 4卡 0.04ms， 40us；
    main()

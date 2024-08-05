import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def measure_latency(rank, size):
    tensor_size = 1024 * 1024  # 1MB tensor
    tensor = torch.rand(tensor_size).cuda(rank)
    
    # Warm-up
    if rank % 2 == 0:
        dist.send(tensor, dst=(rank + 1) % size)
        dist.recv(tensor, src=(rank + 1) % size)
    else:
        dist.recv(tensor, src=(rank - 1 + size) % size)
        dist.send(tensor, dst=(rank - 1 + size) % size)

    # Measure latency
    torch.cuda.synchronize(rank)
    start_time = time.time()

    if rank % 2 == 0:
        dist.send(tensor, dst=(rank + 1) % size)
        dist.recv(tensor, src=(rank + 1) % size)
    else:
        dist.recv(tensor, src=(rank - 1 + size) % size)
        dist.send(tensor, dst=(rank - 1 + size) % size)

    torch.cuda.synchronize(rank)
    end_time = time.time()
    latency = (end_time - start_time) / 2  # round-trip time divided by 2

    print(f"Rank {rank} latency: {latency * 1000:.2f} ms")

def init_process(rank, size, fn, backend='nccl'):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=size)
    torch.cuda.set_device(rank)
    fn(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    size = 8  # Number of GPUs
    mp.spawn(init_process, args=(size, measure_latency), nprocs=size, join=True)
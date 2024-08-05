import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def p2p_communication(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}')
    
    if rank == 0:
         # Warmup
        tensor = torch.zeros(1, device=device)
        for target in range(1, world_size):
            dist.send(tensor, dst=target)
            # dist.recv(tensor, src=target)

        # 同步所有进程，确保warmup完成
        dist.barrier()
        
        # GPU0 发送张量到其他 GPU，并测量时间
        for target in range(1, world_size):
            tensor = torch.ones(1, device=device) * target  # 一个较大的张量，用于通信测试
            start_time = time.time()
            dist.send(tensor, dst=target)
            torch.cuda.synchronize(device)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print(f'P2P communication from GPU0 to GPU{target} took {elapsed_time*1000:.3f} ms')
    else:
        tensor = torch.zeros(1, device=device)

        # Warmup
        dist.recv(tensor, src=0)
        # dist.send(tensor, dst=0)
        
        # 同步所有进程，确保warmup完成
        dist.barrier()


        # 其他 GPU 从 GPU0 接收张量
        dist.recv(tensor, src=0)
        torch.cuda.synchronize(device)
        print(f'Rank {rank} has tensor {tensor} after p2p communication')

    cleanup()

def main():
    world_size = 4  # 使用8个 GPU
    mp.spawn(p2p_communication, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

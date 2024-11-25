import torch

def merge_qkv(hidden_dim, q_dim, k_dim, v_dim, dtype, device):
    print(f"hidden_dim: {hidden_dim}, q_dim: {q_dim}, k_dim: {k_dim}, v_dim: {v_dim}, dtype: {dtype}, device: {device}")
    # 定义输入矩阵 A 和权重矩阵 W1, W2, W3
    A = torch.randn(hidden_dim, hidden_dim).to(dtype=dtype, device=device)  # 示例 2 维矩阵
    W1 = torch.randn(hidden_dim, q_dim).to(dtype=dtype, device=device)
    W2 = torch.randn(hidden_dim, k_dim).to(dtype=dtype, device=device)
    W3 = torch.randn(hidden_dim, v_dim).to(dtype=dtype, device=device)

    # 分别进行矩阵乘法
    result1 = torch.matmul(A, W1)
    result2 = torch.matmul(A, W2)
    result3 = torch.matmul(A, W3)

    # 拼接三个结果
    result_combined = torch.cat((result1, result2, result3), dim=1)

    # 将三个权重矩阵拼接成一个权重矩阵
    W_combined = torch.cat((W1, W2, W3), dim=1)

    # 进行一次矩阵乘法
    result_single = torch.matmul(A, W_combined)

    # 比较两个结果是否一致
    if torch.equal(result_combined, result_single):
        print("三个矩阵乘法结果的拼接与一次矩阵乘法的结果一致")
    else:
        print("三个矩阵乘法结果的拼接与一次矩阵乘法的结果不一致")

    print(torch.allclose(result_combined, result_single))
    print ((torch.abs(result_combined-result_single) > 0.01).sum())
    print ((torch.abs(result_combined-result_single) > 0.1).sum())


merge_qkv(4096, 4096, 1024, 1024, torch.float32, 'cuda:0')
merge_qkv(4096, 4096, 1024, 1024, torch.bfloat16, 'cuda:0')
merge_qkv(4096, 4096, 1024, 1024, torch.float32, 'cpu')
merge_qkv(4096, 4096, 1024, 1024, torch.bfloat16, 'cpu')
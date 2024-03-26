import torch

def embedding_lookup(input_ids, embedding_weights):
    """
    手动实现嵌入查找。
    
    参数:
    - input_ids: 输入序列的整数索引，形状为 [batch_size, seq_length]
    - embedding_weights: 嵌入矩阵的权重，形状为 [vocab_size, embedding_dim]
    
    返回:
    - 嵌入后的序列，形状为 [batch_size, seq_length, embedding_dim]
    """
    # 利用torch的高级索引从embedding_weights中选取input_ids对应的向量
    embedded = embedding_weights[input_ids]
    return embedded


if __name__ == '__main__':
        
    # 示例使用
    batch_size = 4
    seq_length = 32
    vocab_size = 32000  # 假设词汇表大小为32000
    embedding_dim = 4096  # 假设嵌入维度为4096

    # 模拟嵌入权重
    embedding_weights = torch.rand(vocab_size, embedding_dim)

    # 生成随机输入序列的整数索引
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

    # 执行嵌入查找
    embedded_input = embedding_lookup(input_ids, embedding_weights)

    print(embedded_input.shape)  # 输出：torch.Size([batch_size, seq_length, embedding_dim])

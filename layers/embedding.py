import torch


class Embedding():
    def __init__(self, embedding_weights):
        self.embedding_weights = embedding_weights
        self.cache = None

    def forward(self, input_ids):
        """
        手动实现嵌入查找。
        
        参数:
        - input_ids: 输入序列的整数索引，形状为 [batch_size, seq_length]
        - embedding_weights: 嵌入矩阵的权重，形状为 [vocab_size, embedding_dim]
        
        返回:
        - 嵌入后的序列，形状为 [batch_size, seq_length, embedding_dim]
        """
        # 利用torch的高级索引从embedding_weights中选取input_ids对应的向量
        embedded = self.embedding_weights[input_ids]
        self.cache = input_ids
        return embedded

    def backward(self, dy):
        input_ids = self.cache
        grad_embedding_weights = torch.zeros_like(self.embedding_weights)
        emb_dim = self.embedding_weights.size(1)
        # version1:按索引并行累加梯度
        grad_embedding_weights.index_add_(0, input_ids.view(-1), dy.view(-1, emb_dim))
        """
        version2:循环查找索引累加梯度
        
        for i in range(input_ids.shape[0]):
            for j in range(input_ids.shape[1]):
                idx = input_ids[i, j]
                grad_embedding_weights[idx] += dy[i, j]
        """
        return grad_embedding_weights


def test_embedding_lookup_manual_class():
    # 嵌入查找反向传播比较：手写的反向实现与pytorch自带的自动求导
    batch_size = 4
    seq_len = 1024
    vocab_size = 32000
    embedding_dim = 4096

    # 随机生成输入ID和嵌入矩阵
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    embedding_weights = torch.randn(vocab_size, embedding_dim, requires_grad=True)

    # 定义输出的grad
    dy = torch.randn(batch_size, seq_len, embedding_dim)

    # 前向传递
    embedding = Embedding(embedding_weights)
    embedded = embedding.forward(input_ids)

    # reference 的backward结果
    embedded.backward(dy, retain_graph=True)
    dx_ref = embedding_weights.grad.clone()
    embedding_weights.grad.zero_()  # 重置梯度

    # manual backward的结果
    dx_manual = embedding.backward(dy)
    # 最小可通过误差：1e-6
    print(torch.testing.assert_close(dx_ref, dx_manual, rtol=1e-6, atol=1e-6))


if __name__ == '__main__':
        
    # 示例使用
    batch_size = 4
    seq_length = 32
    vocab_size = 32000  # 假设词汇表大小为32000
    embedding_dim = 4096  # 假设嵌入维度为4096

    test_embedding_lookup_manual_class()

import torch
import torch.nn.functional as F


class CrossEntropy:
    def __init__(self, reduction='mean', ignore_index=-100):
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.cache = None 
    def forward(self, input, target):
        """
        Args:
            input: Tensor of shape (N, C) where N is the batch size and C is the number of classes.
            target: Tensor of shape (N,) where each value is the class index (0 <= target[i] < C) for each example in the batch.

        Returns:
            loss: The scalar loss value.
        """
        batch_size = input.size(0)
        # Apply softmax to get probabilities
        softmax = F.softmax(input, dim=-1)

        # Convert target to one-hot encoding
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        self.cache = softmax
        
        # Calculate log of softmax
        log_softmax = torch.log(softmax)

        # Calculate cross entropy loss
        loss = -torch.sum(target_one_hot * log_softmax)
        if self.reduction == 'mean':
            loss = loss / batch_size

        return loss

    def backward(self, target, loss):
        """
        Args:
            input: Tensor of shape (N, C) where N is the batch size and C is the number of classes.
            target: Tensor of shape (N,) where each value is the class index (0 <= target[i] < C) for each example in the batch.

        Returns:
            grad_input: Gradient of the loss with respect to input.
        """
        softmax = self.cache
        batch_size = target.size(0)

        # Convert target to one-hot encoding
        target_one_hot = torch.zeros_like(softmax)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # Gradient of the loss w.r.t input
        grad_input = (softmax - target_one_hot)
        
        if self.reduction == 'mean':
            grad_input = grad_input / batch_size

        return grad_input


def test_cross_entropy_manual_class(reduction='mean'):
    # Example usage
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]], requires_grad=True)  # Example input tensor (logits)
    target = torch.tensor([2, 1])  # Example target tensor (class indices)

    cross_entropy_manual = CrossEntropy(reduction=reduction)
    # Forward pass using custom implementation
    loss = cross_entropy_manual.forward(input, target)

    # Backward pass using custom implementation
    grad_input = cross_entropy_manual.backward(target, loss)

    # Forward pass using PyTorch's built-in function
    official_loss = F.cross_entropy(input, target, reduction=reduction)

    # Backward pass using PyTorch's built-in function
    official_loss.backward()
    official_grad_input = input.grad

    # Compare the results
    print(torch.testing.assert_close(loss, official_loss))
    print(torch.testing.assert_close(grad_input, official_grad_input))

if __name__ == "__main__":
    test_cross_entropy_manual_class()
    test_cross_entropy_manual_class(reduction='sum')
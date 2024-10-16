import torch
import torch.nn.functional as F
import torch.nn as nn


class CrossEntropy:
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    def __init__(self, reduction='mean', ignore_index=-100, weight=None):
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.cache = None
        self.weight = weight

    def forward(self, input, target):
        """
        Args:
            input: Tensor of shape (N, C)
                    N is the batch size
                    C is the number of classes.
            target: Tensor of shape (N,) where each value is the class index (0 <= target[i] < C) for each example in the batch.

        Returns:
            loss: The scalar loss value.
        """
        batch_size = input.size(0)

        # Apply softmax to get probabilities
        softmax = F.softmax(input, dim=-1)
        
        # Convert target to one-hot encoding
        # e.g.: 
        #   N = 3, C = 4
        #   target = [2, 1, -100] -> 
        #   target_one_hot = [[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        
        # Create mask for ignore_index
        valid_mask = (target != self.ignore_index)
        
        # Convert target to one-hot encoding, ignoring ignored indices
        target_one_hot = torch.zeros_like(input)
        valid_target = target[valid_mask]  # Only consider valid targets
        target_one_hot[valid_mask] = torch.zeros_like(input[valid_mask])  # Set ignored rows to zeros
        target_one_hot[valid_mask] = target_one_hot[valid_mask].scatter_(1, valid_target.unsqueeze(1), 1)

        self.cache = softmax
        
        # Calculate log of softmax
        log_softmax = torch.log(softmax)

        # If weight is provided, (and reduction is not mean!!!)  apply it to the target one-hot encoding
        if self.reduction == 'mean' and self.weight is not None:
            weight = self.weight.view(1, -1)  # Reshape weight to (1, C)
            target_one_hot = target_one_hot * weight
        else:
            weight = torch.ones_like(input)  # Default weight is 1

        # Calculate cross entropy loss
        loss = -torch.sum(target_one_hot * log_softmax)
        if self.reduction == 'mean':
            loss = loss / weight[valid_mask].sum()  # Only scale by the number of valid samples
            
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

        # Create mask for ignore_index
        valid_mask = (target != self.ignore_index)

        # Convert target to one-hot encoding, ignoring ignored indices
        target_one_hot = torch.zeros_like(softmax)
        valid_target = target[valid_mask]
        target_one_hot[valid_mask] = target_one_hot[valid_mask].scatter_(1, valid_target.unsqueeze(1), 1)

        # If weight is provided, (and reduction is not mean!!!)  apply it to the target one-hot encoding
        if self.reduction == 'mean' and self.weight is not None:
            weight = self.weight.view(1, -1)  # Reshape weight to (1, C)
            target_one_hot = target_one_hot * weight
        else:
            weight = torch.ones_like(input)  # Default weight is 1
            
        # Gradient of the loss w.r.t input, ignoring ignored indices
        grad_input = softmax.clone()
        grad_input[valid_mask] -= target_one_hot[valid_mask]
        grad_input[~valid_mask] = 0  # Set gradient to zero for ignored indices

        if self.reduction == 'mean':
            grad_input = grad_input / weight[valid_mask].sum()  # Only scale by the number of valid samples

        return grad_input


def test_cross_entropy_manual_class(reduction='mean'):
    # Example usage without ignore_index
    print(f"=====================Testing ignore_index case: No, reduction: {reduction}=====================")
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
    print("Loss comparison:", torch.testing.assert_close(loss, official_loss))
    print("Gradient comparison:", torch.testing.assert_close(grad_input, official_grad_input))

    # Example usage with ignore_index
    print(f"=====================Testing ignore_index case: Yes, reduction: {reduction}=====================")
    input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], requires_grad=True)  # Example input tensor (logits)
    target = torch.tensor([2, -100, 1])  # Example target tensor with ignore_index (-100)
    weight = torch.FloatTensor([0.5, 1.0, 2.0])  # Example class weights

    cross_entropy_manual = CrossEntropy(reduction=reduction, ignore_index=-100, weight=weight)
    # Forward pass using custom implementation
    loss = cross_entropy_manual.forward(input, target)

    # Backward pass using custom implementation
    grad_input = cross_entropy_manual.backward(target, loss)

    # Forward pass using PyTorch's built-in function
    official_loss = F.cross_entropy(input, target, reduction=reduction, ignore_index=-100, weight=weight)

    # Backward pass using PyTorch's built-in function
    official_loss.backward()
    official_grad_input = input.grad

    # Compare the results
    print("Loss comparison with ignore_index:", torch.testing.assert_close(loss, official_loss))
    print("Gradient comparison with ignore_index:", torch.testing.assert_close(grad_input, official_grad_input))


if __name__ == "__main__":
    test_cross_entropy_manual_class()
    test_cross_entropy_manual_class(reduction='sum')
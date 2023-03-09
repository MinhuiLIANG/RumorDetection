import torch
import torch.nn.functional as F

# Example of target with class indices
input = torch.rand(16, 2, requires_grad=True)
target = torch.ones(16)
target = target.long()
loss = F.cross_entropy(input, target)
loss.backward()
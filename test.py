import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)

print(input)
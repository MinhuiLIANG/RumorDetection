import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.rand(4,1,1).to(device)
unimodal_weight = 0.5 * (1 - a).to(device)

print(a)
print(unimodal_weight)

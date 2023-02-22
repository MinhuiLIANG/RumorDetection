import torch

Threshold = 0.5
acc_calcu = torch.rand((16))
labels_tensor = torch.ones((16))

print(acc_calcu)
print(labels_tensor)

acc_calcu[acc_calcu > Threshold] = 1
acc_calcu[acc_calcu < Threshold] = 0

print(acc_calcu)

acc_num = torch.eq(acc_calcu, labels_tensor).sum()

a = float(acc_num + 3)

print(acc_num)
print('a', a)

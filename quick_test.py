import torch

w = torch.tensor([[1, 2], [3, 4], [5, 6]])
x = torch.tensor([[-2, 3]]).T
print(w@x)
print((w@x).sum())
print(w.sum(dim=0)@x)

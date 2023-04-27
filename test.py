import torch
a = torch.rand(2, 2, requires_grad=True) # turn on autograd
print("a : ", a)

b = a.clone()
print("b : ", b)

c = a.detach().clone()
print("c : ", c)
print("a : ", a)

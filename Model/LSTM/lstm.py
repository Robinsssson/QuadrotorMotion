# -*- coding:UTF-8 -*-
import torch

x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64, requires_grad=True)
y = x ** 2
print(y)
print(x)
y.backward(torch.ones_like(y))
print(x.grad)
print(y)
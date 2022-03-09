import torch
from torch.functional import F
from torch import tensor

mins = tensor([[0, 1, 0, 1, 4],
               [0, 0, 2, 2, 4],
               [0, 1, 2, 1, 2]])

mins = mins.view(3 * 5)
a = torch.zeros((3 * 5, 5))

a[torch.arange(15), mins] = 1
a = a.view((3,5,5))

print(a)
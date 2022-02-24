import torch
from torch.functional import F
from torch import tensor

a = tensor([4.0898e+01,  1.9696e+02, -8.0618e+01,  1.8301e+02, -4.9441e+01])

# s = tensor([0.0000e+00, 1.0000e+00, 0.0000e+00, 0, 0.0000e+00])
s = F.softmax(a, dim=-1)
print(s)
print(F.log_softmax(a, dim=-1))
print(s.log())

print(s.multinomial(2, False))

from torch import tensor


a = tensor([[ 0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1.],
        [ 0.,  0.,  1.,  0.,  0., -1., -1., -1., -1., -1.],
        [ 0.,  1.,  0.,  0.,  0., -1., -1., -1., -1., -1.]])

print(1 - a[:, :5])
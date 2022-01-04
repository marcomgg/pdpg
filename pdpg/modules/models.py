import torch.nn as nn
import torch
from torch.distributions import MultivariateNormal, Categorical
import numpy as np

activations = dict(relu=nn.ReLU, tanh=nn.Tanh, sigmoid=nn.Sigmoid)


class OrthoInit:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, m):
        if type(m) == nn.Linear:
            weights = m.weight
            #lasagne ortho init for tf
            shape = tuple(weights.shape)
            if len(shape) == 2:
                flat_shape = shape
            elif len(shape) == 4: # assumes NHWC
                flat_shape = (np.prod(shape[:-1]), shape[-1])
            else:
                raise NotImplementedError
            a = np.random.normal(0.0, 1.0, flat_shape)
            u, _, v = np.linalg.svd(a, full_matrices=False)
            q = u if u.shape == flat_shape else v # pick the one with the correct shape
            q = q.reshape(shape)
            weights.data = torch.Tensor((self.scale * q[:shape[0], :shape[1]]).astype(np.float32))

            m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self, input_size, output_size, num_hidden, hidden_sizes, activation='tanh'):
        super().__init__()

        self.m = nn.Sequential()
        act = activations[activation]

        def dense(in_features, out_features, bias=True):
            if activation == 'relu':
                return nn.Sequential(nn.Linear(in_features, out_features, bias), act(True))
            return nn.Sequential(nn.Linear(in_features, out_features, bias), act())

        in_size = input_size
        for l in range(num_hidden):
            out_size = hidden_sizes[l] if not isinstance(hidden_sizes, int) else hidden_sizes
            self.m.add_module(f"Hidden {l}", dense(in_size, out_size))
            in_size = out_size
        self.m.add_module("Output", nn.Linear(in_size, output_size, True))

    def forward(self, x):
        return self.m(x)

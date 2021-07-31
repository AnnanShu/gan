from typing import ForwardRef
from numpy.core.defchararray import center
from numpy.lib.polynomial import RankWarning
import torch 
import torch.nn.functional as F 
import torch.nn as nn 
from torch import nn, optim, autograd, rand
import numpy as np 
import random 
import visdom


torch.cuda.is_available()

class Generator(nn.module):
    def __init__(self, h_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim), 
            nn.ReLU(True), 
            nn.Linear(h_dim, h_dim), 
            nn.ReLU(True), 
            nn.Linear(h_dim, h_dim), 
            nn.ReLU(True), 
            nn.Linear(h_dim, 2), 
        )

    def forward(self, z):
        output = self.net(z)
        return output

def data_generator(batch_size=32):
    """
    8-gaussian mixture model 
    : return 
    """
    scale = 2. 
    centers = [
        (1, 0), 
        (-1, 0), 
        (0, 1), 
        (0, -1), 
        (1. / np.sqrt(2), 1. / np.sqrt(2)), 
        (1. / np.sqrt(2), -1. / np.sqrt(2)), 
        (-1. / np.sqrt(2), 1. / np.sqrt(2)), 
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]

    while True:
        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.02
            center = random.choice(centers)
            # N(0, 1) + center x1/x2
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point) 

        dataset = np.array(dataset).astype(np.float32) 
        dataset /= np.sqrt(2)
        yield dataset


def main():
    torch.manual_seed(23)
    np.random.seed(23) 
    data_iter = data_generator()
    x = next(data_iter) 
    print(x) 
    return x


if __name__ == '__main__':
    main()
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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
MAX_EPOCH = 50000
BATCH_SIZE = 32
LEARNING_RATE = 5e-4


h_dim = 400
class Generator(nn.Module):
    def __init__(self) -> None:
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

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim), 
            nn.ReLU(True), 
            nn.Linear(h_dim, h_dim), 
            nn.ReLU(True), 
            nn.Linear(h_dim, h_dim), 
            nn.ReLU(True), 
            nn.Linear(h_dim, 1), 
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

def data_generator(batch_size=BATCH_SIZE):
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
        for _ in range(batch_size):
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

    G = Generator().to(device)
    D = Discriminator().to(device)

    optim_G = optim.Adam(G.parameters(), lr=LEARNING_RATE)
    optim_D = optim.Adam(D.parameters(), lr=LEARNING_RATE)

    for epoch in range(MAX_EPOCH):

        # train Discriminator firstly
        for _ in range(5):
            # 1.1 train on real data 
            xr = next(data_iter) 
            xr= torch.from_numpy(x).to(device) 
            # [b, 2] -> [b, 1]
            predr = D(x) 
            lossr = predr.mean()

            # 1.2 train on fake data 
            # [b, ]
            z = torch.randn(BATCH_SIZE, 3).to(device) 
            xf = G(z).detach()  # tf.stop_gradient() 
            predf = D(xf)
            lossf = predf.mean()

            # aggregate all 
            loss_D = lossr + lossf 

            # optimizer
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
        # train Generator

        for _ in range(5):
            z = torch.randn(BATCH_SIZE, 2).to(device)
            xf = G(z) 
            predf = D(xf) 
            # max perdf.mean() 
            loss_G = -predf.mean()
            # optimize
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
    

    


if __name__ == '__main__':
    main()
import numpy as np
import torch
import torch.nn as nn
from utils import Arguments, load_data

from Models.MNIST import MNIST_paper





class Solver:
    def __init__(self, args:Arguments):
        self.args:Arguments = args

        self.model = MNIST_paper()

        self.loss = nn.MSELoss()#.cuda()
        self.data = load_data(args)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          betas=(0.5,0.999))
        
    
    def train(self):
        pass









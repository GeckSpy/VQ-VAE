import numpy as np
import torch
import torch.nn as nn

from utils import Arguments, load_data

from Models.MNIST import MNIST_paper





class Solver:
    def __init__(self, args:Arguments):
        self.args:Arguments = args

        self.model = MNIST_paper()

        self.loss = nn.MSELoss().cuda()
        self.data, self.data_loader = load_data(args)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          betas=(0.5,0.999))
    

    
    def train(self):
        for epoch in range(self.args.epoches):
            reconstruction_losses = []
            sgz_e_losses = []
            z_sge_losses = []

            for id, (image, label) in enumerate(self.data):

                X = ???

                X_recon, Z_enc, Z_dec, Z_enc_for_embd = self.model(X)

                reconstruction_loss = self.loss(X_recon, X)
                sgz_e_loss = self.loss(self.model.embeding.weight, Z_enc_for_embd.detach())
                z_sge_loss = self.loss(Z_enc, Z_dec.detach())
                total_loss = reconstruction_loss + sgz_e_loss + z_sge_loss











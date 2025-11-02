import numpy as np
import torch
import torch.nn as nn

from utils import Arguments, load_data
from Models.MNIST import MNIST_paper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Solver:
    def __init__(self, args:Arguments):
        self.args:Arguments = args.copy()

        if self.args.dataset_name == "MNIST":
            self.model = MNIST_paper()
        else:
            assert ValueError("dataset " + self.args.dataset_name + " is not supported.")

        self.loss = nn.MSELoss().to(device)
        self.data, self.data_loader = load_data(args)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          betas=(0.5,0.999))
    


    def train(self):
        for epoch in range(self.args.epoches):
            reconstruction_losses = []
            sgz_e_losses = []
            z_sge_losses = []

            for _, (images, _) in enumerate(self.data):

                X = images.to(device)
                X_recon, Z_enc, Z_dec, Z_enc_for_embd = self.model(X)

                reconstruction_loss = self.loss(X_recon, X)
                sgz_e_loss = self.loss(self.model.embeding.weight, Z_enc_for_embd.detach())
                z_sge_loss = self.loss(Z_enc, Z_dec.detach())
                total_loss = reconstruction_loss + sgz_e_loss + self.args.beta*z_sge_loss

                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                Z_enc.backward(self.model.grad_for_encoder)
                self.optimizer.step()

                reconstruction_losses.append(reconstruction_loss)
                sgz_e_losses.append(sgz_e_loss)
                z_sge_losses.append(z_sge_loss)











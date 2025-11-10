import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import Arguments, load_data
from Models.MNIST import MNIST_paper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#________________ Solver Class ________________________________
class Solver:
    def __init__(self, args:Arguments):
        self.args:Arguments = args.copy()

        if self.args.dataset_name == "MNIST":
            self.model = MNIST_paper(k_dim=args.k_dim, z_dim=args.z_dim)
        else:
            assert ValueError("dataset " + self.args.dataset_name + " is not supported.")
        self.model.to(device)

        self.loss = nn.MSELoss().to(device)
        self.data, self.data_loader = load_data(args)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          betas=(0.5,0.999))
    


    def train(self):
        """
        Training the model.
        """
        for epoch in range(self.args.epoches):
            reconstruction_losses = []
            sgz_e_losses = []
            z_sge_losses = []

            for id, (images, _) in enumerate(self.data_loader):
                #print(id, end=" ")

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

                reconstruction_losses.append(reconstruction_loss.item())
                sgz_e_losses.append(sgz_e_loss.item())
                z_sge_losses.append(z_sge_loss.item())
            print(epoch, ":")
            print("   reconstruction losses average:", np.average(reconstruction_losses))
            print("   sgz_e losses average:", np.average(sgz_e_losses))
            print("   z_sge losses average:", np.average(z_sge_losses))






def save_model(path, model, optimizer=None, epoch=None, extra=None):
    """Save state dict and optimizer (optional)."""
    root = "./saves"
    if not os.path.exists(root):
        raise ValueError("Wrong root. Please place yourself on correct root directory.")
    root += "/" + path + ".pth"
    #os.makedirs(os.path.dirname(root), exist_ok=True)

    ckpt = {'model': model.state_dict()}
    if optimizer is not None: ckpt['optimizer'] = optimizer.state_dict()
    if epoch is not None: ckpt['epoch'] = epoch
    if extra is not None: ckpt['extra'] = extra
    torch.save(ckpt, root)


def load_model(path_init, model, optimizer=None, device=None):
    """Load checkpoint into model and optimizer (optional)."""
    path = "./saves/" + path_init + ".pth"
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt.get('epoch', None)
    return epoch


# ______________________________

def train_model(args:Arguments, model_name, save=True):
    solver = Solver(args)
    solver.train()
    if save:
        save_model(model_name, solver.model)
    


def test_model(args:Arguments, model_name, K=1):
    solver = Solver(args)
    load_model(model_name, solver.model)
    solver.model.eval()

    datas, _ = next(iter(solver.data_loader))
    data = datas[np.random.randint(0, datas.shape[0]-1, K)].to(device)


    def show_sample(sample, reconstruction):
        if args.dataset_name=="MNIST":
            fig, axs = plt.subplots(2,K)
            fig.subplots_adjust(hspace=-0.8, wspace=0.1)
            
            for k in range(K):
                axs[0,k].imshow(sample[k,0], cmap="gray")
                axs[1,k].imshow(reconstruction[k,0], cmap="gray")
                axs[0,k].axis("off")
                axs[1,k].axis("off")
                
            plt.show()
    
    
    #with torch.no_grad():
    datas_reconstructed, _, _, _ = solver.model(data)
    #print(data.shape, datas_reconstructed.shape)
    show_sample(data, datas_reconstructed.detach())



args = Arguments(dataset_name="MNIST",
                 epoches=30, learning_rate=1e-4, batch_size=100, beta=0.1,
                 k_dim=128, z_dim=64)
#train_model(args, "MNIST_paper1")
test_model(args, "MNIST_paper1", K=12)

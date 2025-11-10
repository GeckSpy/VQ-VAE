import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import Arguments, load_data
from Models.MNIST import MNIST_paper
from Models.PixelCNN import PixelCNN1D, PIXELCNN2D
from VQ_VAE import Solver, save_model, load_model, show_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_randomly(args:Arguments, model_name, B=2):
    """Generate new samples according to random uniform initial samples.
    
    Note that this is not the correct way to do it."""
    solver = Solver(args)
    load_model(model_name, solver.model)
    solver.model.eval()

    datas, _ = next(iter(solver.data_loader))
    data = datas[np.random.randint(0, datas.shape[0]-1, B)]
    data = torch.rand(data.shape).to(device)
    
    datas_reconstructed, _, _, _ = solver.model(data)
    show_sample(data, datas_reconstructed.detach(), args.dataset_name, B)


def create_data_for_training(solver:Solver):
    datas = {"Z":[], "id":[]}
    for id, (images,_) in enumerate(solver.data_loader):
        X = images.to(device)
        Z_enc, embedded_index = solver.model.forward_pixelCNN(X)
        if solver.args.dataset_name=="MNIST":
            Z_enc = Z_enc.unsqueeze(2)
            embedded_index = embedded_index.unsqueeze(1)

        datas["Z"].append(Z_enc.detach().to(device))
        datas["id"].append(embedded_index.detach().to(device).long())
    return datas


def create_pixel_cnn(args_model:Arguments):
    if args_model.dataset_name=="MNIST":
        pixelcnn = PixelCNN1D(k_dim=args_model.k_dim,
                            z_dim=args_model.z_dim,
                            kernel_size=args_CNN.kernel_size,
                            fm=args_CNN.fm)
        
    elif args_model.dataset_name=="CIF10":
        pixelcnn = PIXELCNN2D(k_dim=args_model.k_dim,
                            z_dim=args_model.z_dim,
                            kernel_size=args_CNN.kernel_size,
                            fm=args_CNN.fm)
    return pixelcnn


def train_CNN(args_model:Arguments, model_name:str,
              args_CNN:Arguments, cnn_model_name:str):
    """
    Train the pixel-CNN model for futur sampling.
    """
    solver = Solver(args_model)
    load_model(model_name, solver.model)
    solver.model.eval()
    criterion = F.cross_entropy
    datas= create_data_for_training(solver)

    pixelcnn = create_pixel_cnn(args_model).to(device)
    optimizer = torch.optim.Adam(pixelcnn.parameters(),
                                 lr=args_CNN.learning_rate,
                                 betas=(0.5, 0.999))
    pixelcnn.train()

    for e in range(args_CNN.epoches):
        for i in range(len(datas["id"])):
            Z_enc = datas["Z"][i].to(device)
            index = datas["id"][i].to(device)

            logits = pixelcnn(Z_enc)
            loss = criterion(logits, index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(e, ":", loss)
    
    save_model(cnn_model_name, pixelcnn)



def show_generated(args:Arguments, sample):
    if args.dataset_name=="MNIST":
        B = sample.shape[0]
        images = sample.view(B,28,28)
        fig,axs = plt.subplots(B)
        for i in range(B):
            axs[i].imshow(images[i], cmap="gray")
            axs[i].axis("off")
        plt.show()



def generate_samples(args_model:Arguments, model_name,
                     args_cnn:Arguments, cnn_model_name,
                     L=1, B=10,
                     temperature=1.0):
    """Generate K new sample thank to pixel CNN"""
    solver = Solver(args_model)
    load_model(model_name, solver.model)
    solver.model.eval() # solver.model is the vq-vae model

    pixelcnn = create_pixel_cnn(args_model)
    load_model(cnn_model_name, pixelcnn)
    pixelcnn.eval()

    emb = solver.model.embd.weight
    indices = torch.zeros(B, L, dtype=torch.long, device=device)
    z_cur = torch.zeros(B, emb.size(1), L, device=device)

    with torch.no_grad():
        for t in range(L):
            logits = pixelcnn(z_cur) #[B, k_dim, L]
            logits_t = logits[:, :, t] / max(temperature, 1e-8) #[B, k_dim]
            probs = F.softmax(logits_t, dim=1)  #[B, k_dim]

            sampled = torch.multinomial(probs, num_samples=1).squeeze(1)  # [B]
            indices[:, t] = sampled
            z_cur[:, :, t] = emb[sampled].to(device) # [B, z_dim, L]

        # Convert to decoder input according to vq_model decoder
        if L == 1:
            z_dec_input = z_cur.squeeze(2)   # [B, z_dim]
            generated = solver.model.decode(z_dec_input)
            show_generated(args_model, generated)
            return generated, indices

        # Case B: decoder expects spatial grid. You must reshape z_cur into grid:
        # e.g., if latent grid is (W, H) and L = W*H, we can reshape:
        #   z_grid = z_cur.permute(0, 2, 1).view(B, H, W, D) -> then permute to desired order
        # This depends on your VQ-VAE decoder type (conv vs MLP). If conv decoder:
        #  z_grid = z_cur.permute(0, 2, 1).view(B, W, H, D).permute(0, 3, 1, 2) -> [B, D, W, H]
        # Then pass to conv decoder: generated = vq_model.decode_from_grid(z_grid)
        #
        # I can't guess exact reshape — adapt below to your decoder expectation.
        #
        # Example (square grid):
        L = z_cur.size(2)
        side = int(L**0.5)
        assert side*side == L, "L must be a square number for this example"
        z_grid = z_cur.permute(0, 2, 1).view(B, side, side, z_dim).permute(0, 3, 1, 2)
        # If your decoder can decode this:
        generated = solver.model.decode_from_grid(z_grid)
        return generated, indices












args_model = Arguments(dataset_name="MNIST",
                 epoches=30, learning_rate=1e-4, batch_size=100, beta=0.1,
                 k_dim=128, z_dim=64)
args_CNN = args_model.copy()
args_CNN.modify(learning_rate=1e-3, kernel_size=3, fm=64, epoches=30)
#generate_randomly(args_model, "MNIST_paper1", B=12)
#train_CNN(args_model, "MNIST_paper1", args_CNN, "CNN_MNIST1")
#generate_samples(args_model, "MNIST_paper1", args_CNN, "CNN_MNIST1", B=10)
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


def generate_randomly(args:Arguments, model_name, K=2):
    """Generate new samples according to random uniform initial samples.
    
    Note that this is not the correct way to do it."""
    solver = Solver(args)
    load_model(model_name, solver.model)
    solver.model.eval()

    datas, _ = next(iter(solver.data_loader))
    data = datas[np.random.randint(0, datas.shape[0]-1, K)]
    data = torch.rand(data.shape).to(device)
    
    datas_reconstructed, _, _, _ = solver.model(data)
    show_sample(data, datas_reconstructed.detach(), args.dataset_name, K)


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







args_model = Arguments(dataset_name="MNIST",
                 epoches=30, learning_rate=1e-4, batch_size=100, beta=0.1,
                 k_dim=128, z_dim=64)
args_CNN = args_model.copy()
args_CNN.modify(learning_rate=1e-3, kernel_size=3, fm=64, epoches=30)
#generate_randomly(args_model, "MNIST_paper1", K=12)
#train_CNN(args_model, "MNIST_paper1", args_CNN, "CNN_MNIST1")
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import Arguments, load_data
from Models.MNIST import MNIST_paper
from Models.PixelCNN import PIXELCNN
from VQ_VAE import Solver, load_model, show_sample

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


def train_CNN(args_model:Arguments, model_name:str, args_CNN:Arguments):
    solver = Solver(args_model)
    load_model(model_name, solver.model)
    solver.model.eval()

    criterion = F.cross_entropy

    datas = {"Z":[], "id":[]}
    for id, (images,_) in enumerate(solver.data_loader):
        X = images.to(device)
        Z_enc, embedded_index = solver.model.forward_pixelCNN(X)
        datas["Z"].append(Z_enc.data)
        datas["id"].append(embedded_index)



args_model = Arguments(dataset_name="MNIST",
                 epoches=30, learning_rate=1e-4, batch_size=100, beta=0.1,
                 k_dim=128, z_dim=64)
args_CNN = args_model.copy()
args_CNN = Arguments(dataset_name="MNIST",
                 epoches=30, learning_rate=1e-3, batch_size=100, beta=0.1,
                 k_dim=128, z_dim=64,
                 kernel_size=3, fm=64)
#generate_randomly(args_model, "MNIST_paper1", K=12)
train_CNN(args_model, "MNIST_paper1", args_CNN)
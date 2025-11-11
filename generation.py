import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import Arguments
from Models.PixelCNN import PixelCNN1D, PIXELCNN2D
from VQ_VAE import Solver, save_model, load_model, show_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_randomly(args:Arguments, model_name, B=2):
    """
    Generate new samples according to random normal initial samples.
    """
    solver = Solver(args)
    load_model(model_name, solver.model) # solver.model is the vq-vae model
    solver.model.eval()

    datas, _ = next(iter(solver.data_loader))
    data = datas[np.random.randint(0, datas.shape[0]-1, B)] # Check data shape
    data = torch.randn(data.shape).to(device) # sample random normal data
    
    datas_reconstructed, _, _, _ = solver.model(data)
    show_sample(data, datas_reconstructed.detach(), args.dataset_name, B)


def create_data_for_training(solver:Solver):
    """
    Create correct data set structure for training.
    """
    datas = {"Z":[], "id":[]}
    for id, (images,_) in enumerate(solver.data_loader):
        X = images.to(device)
        Z_enc, embedded_index = solver.model.forward_pixelCNN(X)

        if solver.args.dataset_name=="MNIST":
            Z_enc = Z_enc.unsqueeze(2)
            embedded_index = embedded_index.unsqueeze(1)
        elif solver.args.dataset_name=="CIFAR10":
            embedded_index = embedded_index.view(Z_enc.size(0),Z_enc.size(2),Z_enc.size(3)).data
            Z_enc = Z_enc.data

        datas["Z"].append(Z_enc.detach().to(device))
        datas["id"].append(embedded_index.detach().to(device).long())
    return datas



def create_pixel_cnn(args_model:Arguments):
    """
    Return correct PixelCNN (1D or 2D) for training.
    """
    if args_model.dataset_name=="MNIST":
        pixelcnn = PixelCNN1D(k_dim=args_model.k_dim,
                            z_dim=args_model.z_dim,
                            kernel_size=args_CNN.kernel_size,
                            fm=args_CNN.fm)
        
    elif args_model.dataset_name=="CIFAR10":
        pixelcnn = PIXELCNN2D(k_dim=args_model.k_dim,
                            z_dim=args_model.z_dim,
                            kernel_size=args_CNN.kernel_size,
                            fm=args_CNN.fm)
    return pixelcnn


def train_CNN(args_model:Arguments, model_name:str,
              args_CNN:Arguments, cnn_model_name:str):
    """
    Train the pixel-CNN model.
    """
    solver = Solver(args_model)
    load_model(model_name, solver.model) # solver.model is the vq-vae model
    solver.model.eval()
    criterion = F.cross_entropy
    datas = create_data_for_training(solver)

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
    """
    Show B samples.
    """
    B = sample.shape[0]
    if args.dataset_name=="MNIST":
        images = sample.view(B,28,28)
        fig,axs = plt.subplots(1,B)
        for i in range(B):
            axs[i].imshow(images[i], cmap="gray")
            axs[i].axis("off")
        plt.show()

    elif args.dataset_name=="CIFAR10":
        sample = (sample+1)/2
        images = sample.permute(0,2,3,1)
        fig,axs = plt.subplots(1,B)
        for i in range(B):
            axs[i].imshow(images[i])
            axs[i].axis("off")
        plt.show()
    




def generate_samples(args_model:Arguments, model_name,
                     args_cnn:Arguments, cnn_model_name,
                     L=1, B=10,
                     temperature=1.0):
    """Generate B new sample thank to pixel CNN model"""
    solver = Solver(args_model)
    load_model(model_name, solver.model) # solver.model is the vq-vae model
    solver.model.eval() 

    pixelcnn = create_pixel_cnn(args_model)
    load_model(cnn_model_name, pixelcnn) # load existing pixelCNN model
    pixelcnn.eval()

    emb = solver.model.embd.weight
    if args_model.dataset_name=="MNIST":
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

            z_dec_input = z_cur.squeeze(2)   # [B, z_dim]
            generated = solver.model.decode(z_dec_input)
        

    elif args_model.dataset_name=="CIFAR10":
        z_dim = args_CNN.z_dim

        sample,_ = next(iter(solver.data_loader))
        Z_enc,_ = solver.model.forward_pixelCNN(sample.to(device))
        _,_, Z_w, Z_h = Z_enc.size()

        rand_idx = torch.multinomial(torch.rand(B*Z_w*Z_h, z_dim),1).squeeze().long().to(device)
        rand_Z = emb[rand_idx].view(-1,Z_w,Z_h,z_dim).permute(0,3,1,2)
        starting_point = 3,0
        with torch.no_grad():
            for i in range(Z_w):
                for j in range(Z_h):
                    if i < starting_point[0] or (i == starting_point[0] and j < starting_point[1]):
                            continue
                    logit = pixelcnn(rand_Z.detach())
                    prob = F.softmax(logit, dim=1).data
                    idx = torch.multinomial(prob[:,:,i,j],1).squeeze()
                    rand_Z[:,:,i,j] = emb[idx]
        generated = solver.model.decode(rand_Z).detach()
    
    show_generated(args_model, generated)










# __________________ Examples _______________________
args_model = Arguments(dataset_name="MNIST",
                 epoches=30, learning_rate=1e-4, batch_size=100, beta=0.1,
                 k_dim=128, z_dim=64)
args_CNN = args_model.copy()
args_CNN.modify(learning_rate=1e-3, kernel_size=3, fm=64, epoches=30)
#generate_randomly(args_model, "MNIST_paper1", B=16)
#train_CNN(args_model, "MNIST_paper1", args_CNN, "CNN_MNIST1")
#generate_samples(args_model, "MNIST_paper1", args_CNN, "CNN_MNIST1", B=20)

args_model = Arguments(dataset_name="CIFAR10",
                 epoches=30, learning_rate=1e-4, batch_size=128, beta=0.25,
                 k_dim=512, z_dim=64)
args_CNN = args_model.copy()
args_CNN.modify(learning_rate=1e-3, kernel_size=3, fm=64, epoches=3)
#train_CNN(args_model,"CIFAR10_paper2", args_CNN, "CNN_CIFAR10")
#generate_samples(args_model, "CIFAR10_paper2", args_CNN, "CNN_CIFAR10", B=10)

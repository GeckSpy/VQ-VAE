import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MNIST_paper(nn.Module):
    # same architecture than paper
    def __init__(self, k_dim:int=10, z_dim:int=64):
        super(MNIST_paper, self).__init__()
        self.k_dim = k_dim
        self.z_dim = z_dim

        # Encoder
        self.encode = nn.Sequential(
            nn.Linear(784,1000),
            nn.ReLU(True),
            nn.Linear(1000,500),
            nn.ReLU(True),
            nn.Linear(500,300),
            nn.ReLU(True),
            nn.Linear(300,self.z_dim),
        )

        # Embedding book
        self.embd = nn.Embedding(self.k_dim,self.z_dim).to(device)

        # Decoder
        self.decode = nn.Sequential(
            nn.Linear(self.z_dim,300),
            nn.LeakyReLU(0.1,True),
            nn.Linear(300,500),
            nn.LeakyReLU(0.1,True),
            nn.Linear(500,1000),
            nn.LeakyReLU(0.1,True),
            nn.Linear(1000,784),
            nn.Tanh()
        )


    def find_nearest(self, query, target, return_index=False):
        """
        Maps query (encoder output) to closest embedding vector
        """
        Q = query.unsqueeze(1).repeat(1,target.size(0),1) # Copy 
        T = target.unsqueeze(0).repeat(query.size(0),1,1) # Copy
        distances = (Q-T).pow(2).sum(2) # computes all pairwise distances
        index = distances.min(1)[1] # .min() resturn (min indices)
        if return_index:
            return index
        else:
            return target[index]


    def hook(self, grad):
        """
        Used for gradient handling trick
        """
        self.grad_for_encoder = grad
        return grad
    

    def forward(self, X):
        """
        X: Pytorch tensor or batch_size x MNIST_size(28x28)
        """
        Z_enc = self.encode(X.view(-1, 784)) # Flatten X and encode it
        Z_emb = self.find_nearest(Z_enc, self.embd.weight) # Nearest-neighbor lookup
        Z_emb.register_hook(self.hook) # For gradient handling trick

        X_recons = self.decode(Z_emb).view(-1, 1, 28, 28)
        Z_enc_for_emb = self.find_nearest(self.embd.weight, Z_enc) # update embedding vectors

        return X_recons, Z_enc, Z_emb, Z_enc_for_emb
    

    def forward_pixelCNN(self, X):
        """
        Return wanted information for training pixelCNN
        """
        Z_enc = self.encode(X.view(-1, 784)) # Flatten X and encode it
        index = self.find_nearest(Z_enc, self.embd.weight, return_index=True)
        return Z_enc, index


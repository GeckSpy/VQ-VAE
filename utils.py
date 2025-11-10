import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class Arguments:
    """Class to deal all training arguments"""
    def __init__(self,
                 epoches=100, learning_rate=2e-4,
                 dataset_name="MNIST",
                 batch_size=100, beta=0.25,
                 k_dim=10, z_dim=64,
                 kernel_size=3, fm=64):
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.beta = beta
        self.k_dim = k_dim
        self.z_dim = z_dim
        self.kernel_size = kernel_size
        self.fm = fm

    def copy(self):
        """Copy the arguments"""
        return Arguments(epoches=self.epoches,
                         learning_rate=self.learning_rate,
                         dataset_name=self.dataset_name,
                         batch_size=self.batch_size,
                         beta=self.beta,
                         k_dim=self.k_dim,
                         z_dim=self.z_dim,
                         kernel_size=self.kernel_size,
                         fm=self.fm)

    def modify(self,
                 epoches=None, learning_rate=None,
                 dataset_name=None,
                 batch_size=None, beta=None,
                 k_dim=None, z_dim=None,
                 kernel_size=None, fm=None):
        if epoches!=None: self.epoches=epoches
        if learning_rate!=None: self.learning_rate=learning_rate
        if dataset_name!=None: self.dataset_name=dataset_name
        if batch_size!=None: self.batch_size=batch_size
        if beta!=None: self.beta=beta
        if k_dim!=None: self.k_dim=k_dim
        if z_dim!=None: self.z_dim=z_dim
        if kernel_size!=None: self.kernel_size=kernel_size
        if fm!=None: self.fm=fm




def load_data(args:Arguments, force_dowload=False):
    """Load data depending of an Argument class
    
    Return the data and the loader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    root = "./Datasets/"
    if (not os.path.exists(root + args.dataset_name)) and (not force_dowload):
        error_str = "path to directory dataset: '" + root+args.dataset_name + "' does not exist\nPlease download dataset first"
        raise ValueError(error_str)

    if args.dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,),(0.5,)),
        ])
        num_workers = 4
        data = MNIST(root=root,
                     train=True,
                     transform=transform,
                     download=True)
        

    loader = DataLoader(data,
                        batch_size=args.batch_size,
                        num_workers=num_workers,
                        shuffle=True,
                        drop_last=False)
        
    return data, loader
        



def test():
    arg = Arguments(epoches=100, learning_rate=1e-3, dataset_name="MNIST", batch_size=100, beta=1)
    data, loader = load_data(arg)

    print(type(loader))
    for id, (images, labels) in enumerate(loader):
        print(images.shape, labels.shape)
        break

#test()
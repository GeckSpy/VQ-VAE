import os
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from dataclasses import dataclass, replace

@dataclass
class Arguments:
    epoches: int = 100
    learning_rate: float = 2e-4
    dataset_name: str = "MNIST"
    batch_size: int = 100
    beta: float = 0.25
    k_dim: int = 10
    z_dim: int = 64
    kernel_size: int = 3
    fm: int = 64

    def copy(self):
        """Return a copy of the class"""
        return replace(self)

    def modify(self, **kwargs):
        """Allow modification of wanted parameters."""
        return replace(self, **kwargs)



def load_data(args:Arguments):
    """
    Load data depending of an Argument class
    
    Return the data and the loader
    """
    root = "./Datasets/"
    if (not os.path.exists(root)): # Check good root folder to not re-download dataset
        error_str = "path to directory dataset: '" + root + "' does not exist\nPlease download dataset first"
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
        
    elif args.dataset_name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        num_workers = 16
        data = CIFAR10(root=root,
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
    """Little test function to show how to handle utils' functions"""
    arg = Arguments(epoches=100, learning_rate=1e-3, dataset_name="MNIST", batch_size=100, beta=1)
    arg.modify(dataset_name="CIFAR10")
    data, loader = load_data(arg)

    print(type(loader))
    for id, (images, labels) in enumerate(loader):
        print(images.shape, labels.shape)

#test()
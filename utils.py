import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class Arguments:
    """Class to deal all training arguments"""
    def __init__(self, epoches, learning_rate, dataset_name, batch_size, beta):
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.beta = beta

    def copy(self):
        """Copy the arguments"""
        return Arguments(epoches=self.epoches,
                         learning_rate=self.learning_rate,
                         dataset_name=self.dataset_name,
                         batch_size=self.batch_size,
                         beta=self.beta)




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
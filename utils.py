import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class Arguments:
    def __init__(self, epoches, learning_rate, dataset_name, batch_size):
        self.learning_rate = learning_rate
        self.epoches = epoches
        self.dataset_name = dataset_name
        self.batch_size = batch_size




def load_data(args:Arguments, force_dowload=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    root = "./Datasets/"
    if (not os.path.exists(root + args.dataset_name)) and (not force_dowload):
        error_str = "path to directory dataset: '" + root+args.dataset_name + "' does not exist\nPlease download dataset first"
        raise ValueError(error_str)

    if args.dataset_name == "MNIST":
        num_workers = 4
        data = MNIST(root=root,
                     train=True,
                     transform=transform,
                     download=force_dowload)
        

    loader = DataLoader(data,
                        batch_size=args.batch_size,
                        num_workers=num_workers,
                        shuffle=True,
                        drop_last=False)
        
    return data, loader
        

#arg = Arguments(epoches=100, learning_rate=1e-3, dataset_name="MNIST", batch_size=100)
#load_data(arg)
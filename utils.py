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




def load_data(args:Arguments):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    root = "./Datasests/"

    if args.dataset_name == "MNIST":
        root += "MNIST"
        if not os.path.exists(root):
            print("path to directory dataset: '", root ,"' does not exist")
            raise ValueError

        return
        data = MNIST(root=root,
                     train=True,
                     transform=transform,
                     download=True)
        loader = DataLoader(data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            drop_last=False)
        
arg = Arguments(epoches=100, learning_rate=1e-3, dataset_name="MNIST", batch_size=100)
load_data(arg)
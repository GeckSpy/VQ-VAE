# VQ-VAE project: re-implementation and results' confirmation

## Objectifs

The objective is to explore the paper called Neural Discrete Representation Learning by Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu published in 30 May 2018 [arxiv link](https://arxiv.org/abs/1711.00937).

More preciselly, the objective is to understand in depht the proposed method, to re-implement using newer version of python and more reable code to be able to reproduce results and confirm (or not) the results shown in the paper.

## Structures

The structure of this project is the following one:
- Folder `Models` contains the VQ-VAEs model for MNIST and CIFAR10 dataset and the PixelCNN models for generating new samples.
- `VQ_VAE.py` is the file containing the main training algorithm and testing for VQ-VAE models.
- `generation.py` is the dile containing the training and testing for generating new samples.
- `utils.py` is a python file full of usefull general funtctions.
- It is assumed to have a folder `Datasets` containing MNIST and CIFA10 dataset download via `torchvision.datasets`.
- It is assumed to have a folder `saves` containing saves of the different models.

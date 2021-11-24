"""Checks if relevant packages are installed."""
import torch
import torchvision
import numpy as np

print("PyTorch version:\t {}".format(torch.__version__))
print("Torchvision version:\t {}".format(torchvision.__version__))

x = torch.randn(3, 4)

if torch.cuda.is_available():
    x = x.cuda()
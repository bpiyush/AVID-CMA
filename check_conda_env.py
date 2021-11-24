"""Checks if relevant packages are installed."""
import torch
import torchvision
import numpy as np

x = torch.randn(3, 4)

if torch.cuda.is_available():
    x = x.cuda()
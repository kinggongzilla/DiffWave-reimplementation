import numpy as np
import torch

def zeroOneNorm(x):
    z = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) # apply the normalization formula
    y = 2 * z - 1 # apply the scaling formula
    return y
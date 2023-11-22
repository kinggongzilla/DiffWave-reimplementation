import numpy as np
import torch

#normlizes x to range -1, 1
def negOneToOneNorm(x):
    z = (x - torch.min(x)) / (torch.max(x) - torch.min(x)) # apply the normalization formula
    y = 2 * z - 1 # apply the scaling formula
    return y
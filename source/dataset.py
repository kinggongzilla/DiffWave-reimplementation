import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from config import SAMPLE_RATE, WITH_CONDITIONING, HOP_LENGTH, CROP_MEL_FRAMES

class LatentsData(Dataset):
    def __init__(self, latents_dir, conditional_dir=None, flatten=True) -> None:
        self.latents_dir = latents_dir
        self.conditional_dir = conditional_dir
        self.latents_files = sorted(os.listdir(self.latents_dir))
        self.spec_files = sorted(os.listdir(self.conditional_dir))
        
    def __len__(self):
        return len(os.listdir(self.latents_dir))

    def __getitem__(self, index):
        latent = np.load(os.path.join(self.latents_dir, self.latents_files[index]), allow_pickle=True)
        if not WITH_CONDITIONING:
            return torch.from_numpy(latent).unsqueeze(0)
        spectrogram = np.load(os.path.join(self.conditional_dir, self.spec_files[index]))[0:1,:,:] #get single channel spectrogram from spectrogram with two channels; slicing [0:1] to preserve dimensions
        # normalize and scale to range (-1 , 1)
        z = (latent - np.min(latent)) / (np.max(latent) - np.min(latent)) # apply the normalization formula
        x = 2 * z - 1 # apply the scaling formula
        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(spectrogram)
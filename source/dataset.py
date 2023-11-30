import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import WITH_CONDITIONING, MAX_SAMPLES
from utils import negOneToOneNorm
from tqdm import tqdm


class LatentsData(Dataset):
    def __init__(self, latents_dir, conditional_dir=None, flatten=True) -> None:
        self.latents_dir = latents_dir
        self.conditional_dir = conditional_dir
        self.latents_files = sorted(os.listdir(self.latents_dir))[0:MAX_SAMPLES]
        self.spec_files = sorted(os.listdir(self.conditional_dir))[0:MAX_SAMPLES]
        self.calculate_dataset_mean_and_std()
        
    def __len__(self):
        return len(self.latents_files)

    def __getitem__(self, index):
        latent = torch.from_numpy(np.load(os.path.join(self.latents_dir, self.latents_files[index]), allow_pickle=True))
        # normalize and scale to range (-1 , 1)
        x = latent / self.total_std
        x = negOneToOneNorm(x)
        if not WITH_CONDITIONING:
            return x.flatten().unsqueeze(0)
        spectrogram = torch.from_numpy(np.load(os.path.join(self.conditional_dir, self.spec_files[index])))[0:1,:,:] #get single channel spectrogram from spectrogram with two channels; slicing [0:1] to preserve dimensions

        return x.flatten().unsqueeze(0), spectrogram
    
    def calculate_dataset_mean_and_std(self):
        total_std = 0
        for index in tqdm(range(len(self))):
            data = np.load(self.latents_dir + '/' + self.latents_files[index])
            data = data.astype(np.float32)
            std = data.std() 
            std = std.astype(np.float16)
            total_std += std
        self.total_std = total_std/len(self)
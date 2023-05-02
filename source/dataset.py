import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from config import SAMPLE_RATE

class ChunkedData(Dataset):

    def __init__(self, audio_dir, conditional_dir=None, max_samples=None) -> None:
        self.audio_dir = audio_dir
        self.conditional_dir = conditional_dir
        self.max_samples = max_samples
        count = 0
        # Iterate directory to find total number of samples in training data
        for path in os.listdir(self.audio_dir):
            if os.path.isfile(os.path.join(self.audio_dir, path)):
                count += 1
        self.length = count


    def __len__(self):
        return self.length if self.max_samples is None or self.max_samples > self.length else self.max_samples

    def __getitem__(self, index):
        audio_file = os.listdir(self.audio_dir)[index]
        #load audio file
        waveform, sample_rate = torchaudio.load(os.path.join(self.audio_dir, audio_file))

        #resample if sample rate is higher than SAMPLE_RATE from config.py
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = waveform[0:1,:] #get single channel waveform from waveform with two channels; slicing [0:1] to preserve dimensions
        
        #load conditioning variable (spectrogram) from .npy numpy file
        conditioning_var = None
        if self.conditional_dir is not None:
            conditional_file = os.listdir(self.conditional_dir)[index]
            conditioning_var = torch.from_numpy(np.load(os.path.join(self.conditional_dir, conditional_file)))
            conditioning_var = conditioning_var[0:1,:, :] #get single channel spectrogram slicing [0:1] to preserve dimensions
            return waveform, SAMPLE_RATE, conditioning_var
        else:
            return waveform, SAMPLE_RATE
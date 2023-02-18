import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from source.config import SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChunkedData(Dataset):

    def __init__(self, audio_dir, conditional_dir=None, max_samples=None) -> None:
        self.audio_dir = audio_dir
        self.conditional_dir = conditional_dir
        self.max_samples = max_samples
        count = 0
        # Iterate directory
        for path in os.listdir(self.audio_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.audio_dir, path)):
                count += 1
        self.length = count


    def __len__(self):
        return self.length if self.max_samples is None or self.max_samples > self.length else self.max_samples

    def __getitem__(self, index):
        #load audio
        audio_file = os.listdir(self.audio_dir)[index]
        waveform, sample_rate = torchaudio.load(os.path.join(self.audio_dir, audio_file))
        if sample_rate > SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = waveform[0:1,:] #get single channel waveform from waveform with two channels; slicing [0:1] to preserve dimensions
        
        #load conditional
        conditional = None
        if self.conditional_dir is not None:
            conditional_file = os.listdir(self.conditional_dir)[index]
            #load .npy file
            conditional = torch.from_numpy(np.load(os.path.join(self.conditional_dir, conditional_file)))
            conditional = conditional[0:1,:, :] #get single channel spectrogram slicing [0:1] to preserve dimensions
            return waveform, SAMPLE_RATE, conditional
        else:
            return waveform, SAMPLE_RATE
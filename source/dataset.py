import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import soundfile
import matplotlib.pyplot as plt
import numpy as np

class ChunkedMusDBHQ(Dataset):

    def __init__(self, audio_dir) -> None:
        self.audio_dir = audio_dir

    def __len__(self):
        count = 0
        # Iterate directory
        for path in os.listdir(self.audio_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.audio_dir, path)):
                count += 1
        return count

    def __getitem__(self, index):
        file = os.listdir(self.audio_dir)[index]
        waveform, sample_rate = torchaudio.load(self.audio_dir + file)
        return waveform, sample_rate
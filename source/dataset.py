import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from source.model_constants import SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ChunkedMusDBHQ(Dataset):

    def __init__(self, audio_dir, max_samples=None) -> None:
        self.audio_dir = audio_dir
        self.max_samples = max_samples
        count = 0
        # Iterate directory
        for path in os.listdir(self.audio_dir):
            # check if current path is a file
            if os.path.isfile(os.path.join(self.audio_dir, path)):
                count += 1
        self.length = count


    def __len__(self):
        return self.length if self.max_samples is None else self.max_samples

    def __getitem__(self, index):
        file = os.listdir(self.audio_dir)[index]
        waveform, sample_rate = torchaudio.load(os.path.join(self.audio_dir, file))
        if sample_rate > SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = waveform[0:1,:] #get single channel waveform from waveform with two channels; slicing [0:1] to preserve dimensions

        return waveform, SAMPLE_RATE
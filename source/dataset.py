import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from config import SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, WITH_CONDITIONING, HOP_LENGTH, CROP_MEL_FRAMES

class ChunkedData(Dataset):

    def __init__(self, audio_dir, conditional_dir=None, max_samples=None) -> None:
        self.audio_dir = audio_dir
        self.conditional_dir = conditional_dir
        self.max_samples = max_samples
        self.audio_files = os.listdir(self.audio_dir)
        self.spec_files = [f+".spec.npy" for f in self.audio_files]
        count = 0
        # Iterate directory to find total number of samples in training data
        for path in os.listdir(self.audio_dir):
            if os.path.isfile(os.path.join(self.audio_dir, path)):
                count += 1
        self.length = count

    def __len__(self):
        return self.length if self.max_samples is None or self.max_samples > self.length else self.max_samples

    def __getitem__(self, index):
        audio_file = self.audio_files[index]
        #load audio file
        waveform, sample_rate = torchaudio.load(os.path.join(self.audio_dir, audio_file))

        #resample if sample rate is higher than SAMPLE_RATE from config.py
        if sample_rate != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = waveform[0:1,:] #get single channel waveform from waveform with two channels; slicing [0:1] to preserve dimensions
        
        #load conditioning variable (spectrogram) from .npy numpy file
        conditioning_var = None
        if self.conditional_dir is not None:
            conditional_file = self.spec_files[index]
            conditioning_var = torch.from_numpy(np.load(os.path.join(self.conditional_dir, conditional_file)))
            conditioning_var = conditioning_var[0:1,:, :SAMPLE_RATE*SAMPLE_LENGTH_SECONDS] #get single channel spectrogram slicing [0:1] to preserve dimensions
            return waveform[:,:SAMPLE_RATE*SAMPLE_LENGTH_SECONDS], SAMPLE_RATE, conditioning_var
        else:
            return waveform, SAMPLE_RATE
        

class Collator:
    def collate(self, minibatch):
        samples_per_frame = HOP_LENGTH
        # create numpy array that will store all audios
        audio_list = []
        conditioner_list = []
        for record in minibatch:
            waveform = record[0][0]
            conditioner = record[2][0].T
            if not WITH_CONDITIONING:
                start = np.random.randint(
                    0, waveform.shape[-1] - SAMPLE_LENGTH_SECONDS * SAMPLE_RATE
                )
                end = start + SAMPLE_LENGTH_SECONDS * SAMPLE_RATE
                waveform = waveform[start:end]
                waveform = np.pad(
                    waveform, (0, (end - start) - len(waveform)), mode="constant"
                )
            else:
                start = np.random.randint(0, conditioner.shape[0] - CROP_MEL_FRAMES)
                end = start + CROP_MEL_FRAMES
                conditioner = conditioner[start:end]

                start *= samples_per_frame
                end *= samples_per_frame
                waveform = waveform[start:end]
                waveform = np.pad(
                    waveform, (0, (end - start) - len(waveform)), mode="constant"
                )
            audio_list.append(waveform)
            conditioner_list.append(conditioner.T)

        audio = torch.from_numpy(np.stack(audio_list))
        if not WITH_CONDITIONING:
            return torch.from_numpy(audio), SAMPLE_RATE, None
        spectrogram = torch.from_numpy(np.stack(conditioner_list))
        # insert dimension at 1 for audio
        # audio = audio.unsqueeze(1)
        # spectrogram = spectrogram.unsqueeze(1)
        return audio, SAMPLE_RATE, spectrogram

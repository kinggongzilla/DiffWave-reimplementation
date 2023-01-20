import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
# from source.model import DiffWave
from source.dataset import ChunkedMusDBHQ
from source.train import train


#TODO: load dataset
path = os.path.join('data', 'chunked_audio')
chunked_data = ChunkedMusDBHQ(audio_dir=path)

trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=2,
    shuffle=True,
    )

#define variance schedule
variance_schedule = torch.linspace(0.001, 0.05, 50)

train(8, 4, trainloader, 1, len(variance_schedule), variance_schedule) #C, num_blocks, trainloader, epochs, timesteps, variance_schedule
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
# from source.model import DiffWave
from source.dataset import ChunkedMusDBHQ
from source.train import train

#clear cuda memory
torch.cuda.empty_cache()

#TODO: load dataset
# path = os.path.join('data', 'chunked_audio')
path = os.path.join('chunked_audio')

if len(sys.argv) > 1:
        path = sys.argv[1]

chunked_data = ChunkedMusDBHQ(audio_dir=path)

trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=4,
    shuffle=True,
    )

#define variance schedule
variance_schedule = torch.linspace(0.001, 0.05, 50)

train(32, 10, trainloader, 1, len(variance_schedule), variance_schedule) #C (num residual channels), num_blocks, trainloader, epochs, timesteps, variance_schedule
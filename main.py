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
data_path = os.path.join('chunked_audio')

if len(sys.argv) > 1:
        data_path = sys.argv[1]

chunked_data = ChunkedMusDBHQ(audio_dir=data_path)

#define variance schedule
EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 2 * 1e-4
NUM_BLOCKS = 8
RES_CHANNELS = 32
TIME_STEPS = 50
VARIANCE_SCHEDULE = torch.linspace(0.001, 0.05, TIME_STEPS)

trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

train(RES_CHANNELS, NUM_BLOCKS, trainloader, EPOCHS, TIME_STEPS, VARIANCE_SCHEDULE)
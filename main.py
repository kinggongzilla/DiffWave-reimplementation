import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from source.model import DiffWave
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
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 2 * 1e-4
NUM_BLOCKS = 10
RES_CHANNELS = 32
TIME_STEPS = 50
VARIANCE_SCHEDULE = torch.linspace(0.001, 0.05, TIME_STEPS)
LAYER_WIDTH = 128

trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train(model, optimizer,trainloader, EPOCHS, TIME_STEPS, VARIANCE_SCHEDULE)
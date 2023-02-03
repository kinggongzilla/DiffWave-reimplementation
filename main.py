import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from source.model import DiffWave
from source.dataset import ChunkedMusDBHQ
from source.train import train
from source.model_constants import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH

#clear cuda memory
torch.cuda.empty_cache()

data_path = os.path.join('chunked_audio')
max_samples = None

#example: python main.py path/to/data 20000
if len(sys.argv) > 1:
    data_path = sys.argv[1]
if len(sys.argv) > 2:
    max_samples = int(sys.argv[2])    

chunked_data = ChunkedMusDBHQ(audio_dir=data_path, max_samples=max_samples)

trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train(model, optimizer,trainloader, EPOCHS, TIME_STEPS, VARIANCE_SCHEDULE)
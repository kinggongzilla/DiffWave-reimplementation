import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb
from source.model import DiffWave
from source.dataset import ChunkedMusDBHQ
from source.train import train
from source.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, TIMESTEP_LAYER_WIDTH, SAMPLE_RATE, MAX_SAMPLES

#clear cuda memory
torch.cuda.empty_cache()

data_path = os.path.join('chunked_audio')

#example: python main.py path/to/data 20000
if len(sys.argv) > 1:
    data_path = sys.argv[1]


wandb.init(project="DiffWave", entity="daavidhauser")

wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_blocks": NUM_BLOCKS,
    "res_channels": RES_CHANNELS,
    "time_steps": TIME_STEPS,
    "variance_schedule": VARIANCE_SCHEDULE,
    "layer_width": TIMESTEP_LAYER_WIDTH,
    "sample_rate": SAMPLE_RATE
} 

chunked_data = ChunkedMusDBHQ(audio_dir=data_path, max_samples=MAX_SAMPLES)

trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, TIMESTEP_LAYER_WIDTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train(model, optimizer,trainloader, EPOCHS, TIME_STEPS, VARIANCE_SCHEDULE)
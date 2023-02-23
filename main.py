import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import wandb
from source.model import DiffWave
from source.dataset import ChunkedData
from source.train import train
from source.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, TIMESTEP_LAYER_WIDTH, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, MAX_SAMPLES, WITH_CONDITIONING, N_MELS

#start with empty cache
torch.cuda.empty_cache()

#default data location
data_path = os.path.join('data/chunked_audio')
conditional_path = os.path.join('data/mel_spectrograms') if WITH_CONDITIONING else None

#example: python main.py path/to/data
if len(sys.argv) > 1:
    data_path = sys.argv[1]

if len(sys.argv) > 2:
    conditional_path = sys.argv[2]

#initialize wandb
wandb.init(
    project="DiffWave", 
    entity="daavidhauser",
    config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_blocks": NUM_BLOCKS,
    "res_channels": RES_CHANNELS,
    "time_steps": TIME_STEPS,
    "variance_schedule": VARIANCE_SCHEDULE,
    "timestep_layer_width": TIMESTEP_LAYER_WIDTH,
    "sample_rate": SAMPLE_RATE,
    "sample_length_seconds": SAMPLE_LENGTH_SECONDS,
    "max_samples": MAX_SAMPLES,
    "with_conditional": WITH_CONDITIONING
    }
)

#initialize dataset
chunked_data = ChunkedData(audio_dir=data_path, conditional_dir=conditional_path, max_samples=MAX_SAMPLES)

#initialize dataloader
trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

#initialize model
model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, WITH_CONDITIONING, N_MELS, layer_width=TIMESTEP_LAYER_WIDTH)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#train model
train(model, optimizer,trainloader, EPOCHS, TIME_STEPS, VARIANCE_SCHEDULE)

#generate a sample directly after training
import sample
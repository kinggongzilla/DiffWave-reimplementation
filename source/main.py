import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from model import DiffWave, LitModel
from dataset import LatentsData
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, TIMESTEP_LAYER_WIDTH, SAMPLE_RATE, MAX_SAMPLES, WITH_CONDITIONING, N_MELS, TRAIN_ON_SUBSAMPLES
from simple_cnn import SimpleCNN

torch.manual_seed(42)

#start with empty cache
torch.cuda.empty_cache()

#set precision
torch.set_float32_matmul_precision('medium')

#print device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

#default data location
data_path = os.path.join('../data/encoded_audio/')
conditional_path = os.path.join('../data/mel_spectrograms') if WITH_CONDITIONING else None
model_checkpoint = None

#example: python main.py path/to/data
if len(sys.argv) > 1:
    data_path = sys.argv[1]

if len(sys.argv) > 2:
    conditional_path = sys.argv[2]

if len(sys.argv) > 3:
    model_checkpoint = sys.argv[3]



#initialize wandb
wandb_logger = WandbLogger(
    project="DiffWave", 
    entity="daavidhauser",
    name="BATCH_SIZE_" + str(BATCH_SIZE) + "_LEARNING_RATE_" + str(LEARNING_RATE) + "_TIME_STEPS_" + str(TIME_STEPS) + "_RES_CHANNELS_" + str(RES_CHANNELS) + "_VARIANCE_SCHEDULE_[" + str(VARIANCE_SCHEDULE[0])  + ", " + str(VARIANCE_SCHEDULE[-1]) + "]" + "_EPOCHS_" + str(EPOCHS),
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
    "max_samples": MAX_SAMPLES,
    "with_conditional": WITH_CONDITIONING
    }
)

checkpoint_callback = ModelCheckpoint(
    monitor='train_loss',
    mode='min',
    dirpath='output/models/',
    filename=f'RES_CH_{RES_CHANNELS}_N_BLOCKS_{NUM_BLOCKS}_DIF_STEPS_{TIME_STEPS}_B_SIZE_{BATCH_SIZE}_LR_{LEARNING_RATE}_EPOCHS_{EPOCHS}_CONDITIONING_{WITH_CONDITIONING}',
    save_top_k=1,
)

#initialize dataset
chunked_data = LatentsData(latents_dir=data_path, conditional_dir=conditional_path)

#initialize dataloader
trainloader = torch.utils.data.DataLoader(
    chunked_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=24, 
    )

#initialize model
model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, WITH_CONDITIONING, N_MELS, layer_width=TIMESTEP_LAYER_WIDTH)
# model = SimpleCNN()
lit_model = LitModel(model)

#train model
trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=EPOCHS, accelerator="auto", devices="auto", precision=16, logger=wandb_logger)

trainer.fit(model=lit_model, train_dataloaders=trainloader, ckpt_path=model_checkpoint)
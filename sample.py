import os
import sys
import numpy as np
import torch
import torchaudio
import wandb
from source.model import DiffWave
from source.config import NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, TIMESTEP_LAYER_WIDTH, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, N_MELS, WITH_CONDITIONING

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "output/models/best_model.pt"
if WITH_CONDITIONING:
    conditioner_file_name = os.listdir("data/mel_spectrograms/")[0] #first file in mel_spectrogram folder
if len(sys.argv) > 1:
    model_path = sys.argv[1]

if len(sys.argv) > 2:
    conditioner_file_name = sys.argv[2]



model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, WITH_CONDITIONING, N_MELS,)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

#load spectrogram
conditioning_var=None
if WITH_CONDITIONING:
    conditioning_var = torch.from_numpy(np.load(os.path.join("data/mel_spectrograms/", conditioner_file_name)))
    conditioning_var = torch.unsqueeze(conditioning_var[0:1, :, :], 0)
noise = torch.randn(1, 1, SAMPLE_RATE*SAMPLE_LENGTH_SECONDS) # batch_size, n_channels, sample length e.g. 22,05KHz * 5000 milliseconds = 5 seconds of noise
y = model.sample(noise, conditioning_var=conditioning_var if model.with_conditioner else None)
for i in range(y.shape[0]): #for each sample in batch
    path = os.path.join("output/samples", f"sample{i}.wav")
    torchaudio.save(path, y[i], SAMPLE_RATE)
    wandb.save(path)
    print('Saved sample to', path)
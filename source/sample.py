import os
import sys
import numpy as np
import torch
import torchaudio
import wandb
from model import DiffWave, LitModel
from config import NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, TIMESTEP_LAYER_WIDTH, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, N_MELS, WITH_CONDITIONING

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#default path to model used for sampling/inference
checkpoint = "./output/models/best_model.ckpt" 

if WITH_CONDITIONING:
    #default to using first file in mel_spectrogram folder as conditioning variable
    conditioner_file_name = os.listdir("data/mel_spectrograms/")[0] 

#get path to model, if given as argument
if len(sys.argv) > 1:
    checkpoint = sys.argv[1]

#get path to conditioning variable file, if given as argument
if len(sys.argv) > 2:
    conditioner_file_name = sys.argv[2]


#load trained model
diffwave = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, WITH_CONDITIONING, N_MELS,)
trained_diffwave = LitModel.load_from_checkpoint(checkpoint, model=diffwave)


# choose your trained nn.Module
model = trained_diffwave.model
model.eval()

#load conditioning variable (spectrogram)
conditioning_var=None
if WITH_CONDITIONING:
    conditioning_var = torch.from_numpy(np.load(os.path.join("data/mel_spectrograms/", conditioner_file_name)))
    conditioning_var = torch.unsqueeze(conditioning_var[0:1, :, :], 0).to(device)

#generate starting noise
noise = torch.randn(1, 1, SAMPLE_RATE*SAMPLE_LENGTH_SECONDS).to(device) # batch_size, n_channels, sample length e.g. 16KHz * 4000 milliseconds = 4 seconds of noise

#get denoised sample
y = model.sample(noise, conditioning_var=conditioning_var if model.with_conditioner else None).to('cpu')

#save audio for each generated sample in batch
for i in range(y.shape[0]):
    #save audio locally
    #use random integer in sample file name, to not accidentally overwrite old generated samples
    random_int = np.random.randint(0, 1000000)
    path = os.path.join("output/samples", f"sample{random_int}.wav") #use random int to make name unique if sample is called multiple times during training
    torchaudio.save(path, y[i], SAMPLE_RATE)

    #save audio to wandb, if wandb is initialized
    if wandb.run is not None:
        wandb.save(path)

    print('Saved sample to', path)
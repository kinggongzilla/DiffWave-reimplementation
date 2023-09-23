import os
import sys
import numpy as np
import torch
import torchaudio
import wandb
from model import DenoisingModel, LitModel
from config import WITH_CONDITIONING

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#default path to model used for sampling/inference
checkpoint = "./output/models/UNET_DIF_STEPS_2_B_SIZE_128_LR_2e-05_EPOCHS_1000_CONDITIONING_True.ckpt" 

if WITH_CONDITIONING:
    #default to using first file in mel_spectrogram folder as conditioning variable
    conditioner_file_name = os.listdir("../data/mel_spectrograms/")[8999]

#get path to model, if given as argument
if len(sys.argv) > 1:
    checkpoint = sys.argv[1]

#get path to conditioning variable file, if given as argument
if len(sys.argv) > 2:
    conditioner_file_name = sys.argv[2]


#load trained model
diffwave = DenoisingModel()
trained_diffwave = LitModel.load_from_checkpoint(checkpoint, model=diffwave).to(device)


# choose your trained nn.Module
model = trained_diffwave.model
model.eval()

#load conditioning variable (spectrogram)
conditioning_var=None
if WITH_CONDITIONING:
    conditioning_var = torch.from_numpy(np.load(os.path.join("../data/mel_spectrograms_unet/", conditioner_file_name)))
    conditioning_var = torch.unsqueeze(conditioning_var[0:1, :, :], 0).to(device)

#generate starting noise
noise = torch.randn(1, 1, 128, 109).to(device) # batch size x channel x flattened latent size
# normed_noise = zeroOneNorm(noise)

#get denoised sample
y = model.sample(noise, conditioning_var=conditioning_var if WITH_CONDITIONING else None).to(device)

#save audio for each generated sample in batch
for i in range(y.shape[0]):
    #save audio locally
    #use random integer in sample file name, to not accidentally overwrite old generated samples
    random_int = np.random.randint(0, 1000000)
    path = os.path.join("output/samples", f"sample{random_int}.wav") #use random int to make name unique if sample is called multiple times during training

    # scale from (-1, 1) to gaussian (for rave latents)
    y = y/y.std()

    # scale from (-1, 1) to (0, 255) for images
    # z = (y + 1) / 2
    # y = z * (255 - 0) + 0 # IMAGE: apply the reverse normalization 
    
    output = y[i].squeeze(0)
    
    np.save(path, output.to('cpu'))

    #save audio to wandb, if wandb is initialized
    if wandb.run is not None:
        wandb.save(path)

    print('Saved sample to', path)
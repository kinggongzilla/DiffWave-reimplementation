import os
import sys
from utils import negOneToOneNorm
sys.path.insert(0, '/home/david/JKU/thesis/DiffWave-reimplementation/source')
import numpy as np
import torch
from model.model import DenoisingModel, LitModel
from config import WITH_CONDITIONING, PRED_NOISE

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#default path to model used for sampling/inference

#currently best diffusion model 23 oct 2023
# checkpoint = "./output/models/2023-10-23_diffusion_with_validation_full_jamendo_continued/UNET_DIF_STEPS_1000_B_SIZE_96_LR_2e-05_EPOCHS_350_CONDITIONING_True.ckpt" 

#currently best oneshot model 23 oct 2023
# checkpoint = "./output/models/2023-10-23_08-25-53/UNET_DIF_STEPS_2_B_SIZE_96_LR_2e-05_EPOCHS_350_CONDITIONING_True.ckpt"

checkpoint = "output/models/2023-11-30_22-44-19/last.ckpt"

if WITH_CONDITIONING:
    #default to using first file in mel_spectrogram folder as conditioning variable
    conditioning_var_path = "../data/jamendo/jamendo_techno_mel_spectrograms/" + sorted(os.listdir("../data/jamendo/jamendo_techno_mel_spectrograms/"))[0]
    target_sample_for_comparison_path = "../data/jamendo/jamendo_techno_encoded_audio/" + sorted(os.listdir("../data/jamendo/jamendo_techno_encoded_audio/"))[0]
    target_sample_for_comparison = negOneToOneNorm(torch.tensor(np.load(target_sample_for_comparison_path)).to(device))


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
    conditioning_var = torch.from_numpy(np.load(conditioning_var_path))
    conditioning_var = torch.unsqueeze(conditioning_var[0:1, :, :], 0).to(device)

#generate starting noise
noise = torch.randn(1, 1, 128, 109).to(device) # batch size x channel x flattened latent size
#get denoised sample
if PRED_NOISE:
    y = model.sample(noise, target_sample_for_comparison if WITH_CONDITIONING else None, conditioning_var=conditioning_var if WITH_CONDITIONING else None).to(device)
else:
    y = model.sample_xt(noise, target_sample_for_comparison if WITH_CONDITIONING else None, conditioning_var=conditioning_var if WITH_CONDITIONING else None).to(device)

#save audio for each generated sample in batch
for i in range(y.shape[0]):
    random_int = np.random.randint(0, 1000000)
    path = os.path.join("output/samples", f"sample{random_int}")
    output = y[i].squeeze(0)
    output = output / output.std()
    np.save(path, output.to('cpu'))
    print('Saved sample to', path)
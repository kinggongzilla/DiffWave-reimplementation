import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import wandb
from config import VARIANCE_SCHEDULE, N_MELS, TIME_STEPS, WITH_CONDITIONING, LEARNING_RATE, SAMPLE_RATE, RES_CHANNELS
import datetime
from unet import UNet
from blocks import input_latent, input_spectrogram, output_latent, input_timestep

# def Conv1d(*args, **kwargs):
#   layer = torch.nn.Conv1d(*args, **kwargs)
#   torch.nn.init.kaiming_normal_(layer.weight)
#   return layer


class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_spectrogram = input_spectrogram(1, 32)
        self.input_timestep = input_timestep(len(VARIANCE_SCHEDULE))
        self.input_latent = input_latent(1, 32)
        self.unet = UNet(64, 64)
        self.output_latent = output_latent(64, 1)

    def forward(self, x, t, conditioning_var):
        c = self.input_spectrogram(conditioning_var)
        t = self.input_timestep(t)
        x = self.input_latent(x)
        x = x + t
        x = torch.cat([x, c,], axis=1)
        x = self.unet(x, t, c)
        x = self.output_latent(x)
        return x



class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def training_step(self, batch, batch_idx):
        #get waveform from (waveform, sample_rate) tuple;

        if WITH_CONDITIONING:
            latent = batch[0] # batch size, channels, length 
        else:
            latent = batch

        #generate noise
        noise = torch.randn(latent.shape)

        #generate random integer between 1 and number of diffusion timesteps
        t = torch.randint(0, TIME_STEPS, (1,))

        #define scaling factors for original waveform and noise

        beta = VARIANCE_SCHEDULE
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        #put all tensors on correct device
        device = self.device
        alpha_cum = alpha_cum.to(device)
        noise = noise.to(device)

        #create noisy version of original waveform
        latent = torch.sqrt(alpha_cum[t])*latent + torch.sqrt(1-alpha_cum[t])*noise

        del alpha_cum, beta, alpha

        conditioning_var = None
        if WITH_CONDITIONING:
            # get conditioning_var (spectrogram) from (waveform, sample_rate, spectrogram) tuple;
            conditioning_var = batch[1] # spectrogram shape: batch size, channels, length

        # predict noise at diffusion timestep t
        y_pred = self.model.forward(latent, t, conditioning_var)

        del t

        #calculate loss and return loss
        batch_loss = F.l1_loss(y_pred, noise)

        del noise

        self.log('train_loss', batch_loss, on_epoch=True)

        return batch_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        return [optimizer], []
    

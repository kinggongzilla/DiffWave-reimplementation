import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import wandb
from config import VARIANCE_SCHEDULE, TIME_STEPS, WITH_CONDITIONING, LEARNING_RATE
import datetime
from unet import UNet
from blocks import input_latent, input_spectrogram, output_latent, input_timestep
from utils import zeroOneNorm

class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        if WITH_CONDITIONING:
            self.input_spectrogram = input_spectrogram(1, 32)
        self.input_timestep = input_timestep(len(VARIANCE_SCHEDULE))
        self.input_latent = input_latent(1, 32)
        if WITH_CONDITIONING:
            self.unet = UNet(64, 64)
        else:
            self.unet = UNet(32, 64)
        self.output_latent = output_latent(64, 1)

    def forward(self, x, t, conditioning_var):
        if WITH_CONDITIONING:
            c = self.input_spectrogram(conditioning_var)
        t = self.input_timestep(t)
        x = self.input_latent(x)
        x = x + t
        if WITH_CONDITIONING:
            x = torch.cat([x, c,], axis=1)
            x = self.unet(x, t, c)
        else:
            x = self.unet(x, t)
        x = self.output_latent(x)
        return x


      #generate a sample from noise input
    def sample(self, x_t, conditioning_var=None):
        with torch.no_grad():

        #sample t-1 sample directly, do not predict noise
            for n in tqdm(range(len(VARIANCE_SCHEDULE) - 2, -1, -1)):
                x_t = self.forward(x_t, torch.tensor(n+1), conditioning_var)

                np.save("output/samples/every_sample_step/every_sample_step_" + str(n), x_t.squeeze(0).squeeze(1).cpu().numpy())

            # beta = self.variance_schedule
            # alpha = 1 - beta
            # alpha_cum = np.cumprod(alpha)

            # #code below actually performs the sampling
            # for n in tqdm(range(len(alpha) - 1, -1, -1)):
            #   c1 = 1 / alpha[n]**0.5 # c1 approaches 1 as timestep gets closer to 0
            #   c2 = beta[n] / (1 - alpha_cum[n])**0.5
            #   x_t = c1 * (x_t - c2 * self.forward(x_t, torch.tensor(n), conditioning_var).squeeze(1))
            #   if n > 0:
            #     noise = torch.randn_like(x_t)
            #     sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
            #     sigma = beta[n]**0.5 #this sigma is optimal if x_0 is from gaussian; see section 3.2 Denoising Diffusion Probabilistic Models
            #     x_t += sigma * noise

            #     #print c1, c2 and sigma every 100 timesteps
            #     if n % 100 == 0:
            #       print("c1: ", c1, "c2: ", c2, "sigma: ", sigma)
        # x_t = torch.clamp(x_t, -1.0, 1.0)

        return x_t.squeeze(1)



class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def training_step(self, batch, batch_idx):
        device = self.device

        if WITH_CONDITIONING:
            latent = batch[0] # batch size, channels, length 
        else:
            latent = batch

        #generate noise
        noise = torch.randn(latent.shape)

        #generate random integer between 1 and number of diffusion timesteps
        t = torch.randint(1, TIME_STEPS, (1,))

        #define scaling factors for original waveform and noise

        beta = VARIANCE_SCHEDULE.to(device)
        # alpha = 1 - beta
        # alpha_cum = np.cumprod(alpha)

        # #put all tensors on correct device
        # alpha_cum = alpha_cum.to(device)
        noise = noise.to(device)

        #create noisy version of original waveform
        # latent = torch.sqrt(alpha_cum[t])*latent + torch.sqrt(1-alpha_cum[t])*noise
        # latent_less_noisy = torch.sqrt(alpha_cum[t-1])*latent + torch.sqrt(1-alpha_cum[t-1])*noise

        noisy_latent = (1- beta[t])*latent + (beta[t])*noise
        less_noisy_latent = (1- beta[t-1])*latent + (beta[t-1])*noise

        # noisy_latent = zeroOneNorm(noisy_latent)
        # less_noisy_latent = zeroOneNorm(less_noisy_latent)


        # del alpha_cum, beta, alpha

        conditioning_var = None
        if WITH_CONDITIONING:
            conditioning_var = batch[1] # spectrogram shape: batch size, channels, length

        # predict noise at diffusion timestep t
        y_pred = self.model.forward(noisy_latent, t, conditioning_var)

        del t

        #calculate loss and return loss
        batch_loss = F.l1_loss(y_pred, less_noisy_latent)

        del noise

        self.log('train_loss', batch_loss, on_epoch=True)

        return batch_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        return [optimizer], []
    

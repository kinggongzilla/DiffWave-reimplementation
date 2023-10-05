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
import math

class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_timestep = input_timestep(TIME_STEPS)
        self.input_latent = input_latent(1, 32)
        if WITH_CONDITIONING:
            self.input_spectrogram = input_spectrogram(1, 32)
            self.unet = UNet(64, 64)
        else:
            self.unet = UNet(32, 64)
        self.output_latent = output_latent(64, 1)

    def forward(self, x, t, conditioning_var):
        t = self.input_timestep(t)
        x = self.input_latent(x)
        x = x + t
        if WITH_CONDITIONING:
            c = self.input_spectrogram(conditioning_var)
            x = torch.cat([x, c,], axis=1)
            x = self.unet(x,)
        else:
            x = self.unet(x,)
        x = self.output_latent(x)
        return x


      #generate a sample from noise input
    def sample(self, x_t, conditioning_var=None):


        gamma = lambda t: simple_linear_schedule(t.item())
        # gamma = lambda t: exponential_schedule(t.item(), tau=0.2)
        # gamma = lambda t: cosine_schedule(t.item(),)

        with torch.no_grad():

            previous_noise = x_t

            #code below actually performs the sampling
            for n in tqdm(range(TIME_STEPS)):
                t_now = torch.tensor(min(1 - n / TIME_STEPS, 0.999)).to(x_t.device)
                t_next = torch.tensor(max(1 - (n+1) / TIME_STEPS, 0.0001)).to(x_t.device)

                # delta_t = t_now - t_next

                # alpha_cum_now = gamma(t_now)
                # alpha_cum_next = gamma(t_next)
                # alpha_now = alpha_cum_now / alpha_cum_next
                # beta_now = 1 - alpha_now


                x_t = x_t / x_t.std(axis=(1,2,3), keepdims=True) if True else x_t
                noise_pred = self.forward(x_t, t_now, conditioning_var).squeeze(1)
                x_t = (x_t - torch.sqrt(1-gamma(t_now)) * noise_pred) / torch.sqrt(gamma(t_now))

                np.save("output/samples/samples_minus_noise/samples_minus_noise_" + str(n), np.clip(x_t.squeeze(0).squeeze(0).cpu().numpy(), -1.0, 1.0))

                # if t_now < 0.70:
                #     break

                if n < TIME_STEPS - 1:
                    if n % 10 == 0:
                        print("t_now: ", t_now)
                        print(f"difference to added noise at step: {n}: ", F.l1_loss(noise_pred, previous_noise))
                    noise = torch.randn_like(x_t)
                    # previous_noise = noise
                    x_t = torch.sqrt(gamma(t_next)) * x_t + torch.sqrt(1 - gamma(t_next)) * noise_pred
                    # sigma = ((1.0 - gamma(t_next)) / (1.0 - gamma(t_now)) * beta_now)**0.5
                    # x_t += sigma * noise

                np.save("output/samples/every_sample_step/every_sample_step_" + str(n), np.clip(x_t.squeeze(0).squeeze(0).cpu().numpy(), -1.0, 1.0))

            x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t.squeeze(1)





class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_per_diffusion_step = np.zeros((10,))
    def training_step(self, batch, batch_idx):

        device = self.device

        if WITH_CONDITIONING:
            x_0 = batch[0] # batch size, channels, length 
        else:
            x_0 = batch

        scale = 1 #TODO: adjust later?

        #linear variance schedule gamma function; can be changed to cos or sigmoid. Look at Importance of Noise Scheduling paper
        gamma = lambda t: simple_linear_schedule(t.item())
        # gamma = lambda t: exponential_schedule(t.item(), tau=0.2)
        # gamma = lambda t: cosine_schedule(t.item(),)

        #generate random integer between 1 and number of diffusion timesteps
        t = torch.rand(1).to(device)
        # t = 0.7 + (t * (1 - 0.7))

        noise = torch.randn(x_0.shape).to(device)

        x_t = torch.sqrt(gamma(t)) * scale * x_0 + torch.sqrt(1-gamma(t)) * noise

        x_t = x_t / x_t.std(dim=(1,2,3), keepdim=True)

        conditioning_var = None
        if WITH_CONDITIONING:
            conditioning_var = batch[1] # spectrogram shape: batch size, channels, length

        # predict noise at diffusion timestep t
        y_pred = self.model.forward(x_t, t, conditioning_var)

        #calculate loss and return loss
        batch_loss = F.l1_loss(y_pred, noise)



        bins = np.arange(0, 1.1, 0.1)
        bin_number = np.digitize(t.item(), bins)
        self.log('train_loss', batch_loss, on_epoch=True)

        # self.log_dict({
        #     'train_loss': batch_loss,
        #     f'diffusion_step_{bin_number}_loss': batch_loss,
        # }, on_epoch=True)

        return batch_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        return [optimizer], []
    

def simple_linear_schedule(t, clip_min=1e-9):
    # A gamma function that simply is 1-t.
    return torch.tensor(np.clip(1 - t, clip_min, 1.))


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.tensor(np.clip(output, clip_min, 1.))

def exponential_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    # A gamma function based on exponential function.
    v_start = math.exp(-start * tau)
    v_end = math.exp(-end * tau)
    output = math.exp(-t * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.tensor(np.clip(output, clip_min, 1.))
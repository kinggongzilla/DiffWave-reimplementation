from typing import Any, Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import wandb
from config import VARIANCE_SCHEDULE, TIME_STEPS, WITH_CONDITIONING, LEARNING_RATE, PRED_NOISE, NOISE_SCHEDULE_FUNC, SCALE, TAU, SAMPLING_ALGORITHM, FIXED_TIMESTEPS, N_MELS, RES_CHANNELS, NUM_BLOCKS, TIMESTEP_LAYER_WIDTH
from model.diffwave import DiffWave
from blocks import input_latent, input_spectrogram, output_layer, input_timestep
import math
from utils import negOneToOneNorm


if NOISE_SCHEDULE_FUNC == 'linear':
    gamma = lambda t: simple_linear_schedule(t.item())
elif NOISE_SCHEDULE_FUNC == 'exp':
    gamma = lambda t: exponential_schedule(t.item(), tau=TAU)
elif NOISE_SCHEDULE_FUNC == 'cos':
    gamma = lambda t: cosine_schedule(t.item(), tau=TAU)

class DenoisingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_timestep = input_timestep()
        self.input_latent = input_latent(1, 32)
        self.diffwave = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, WITH_CONDITIONING, N_MELS, layer_width=TIMESTEP_LAYER_WIDTH)

    def forward(self, x, t, conditioning_var):
        x = self.diffwave(x, t, conditioning_var)
        return x

      #generate a sample from noise input
    def sample(self, x_t, target_sample_for_comparison, conditioning_var=None):

        #TODO: delete later; for testing: instead of noise pass a noisy panda and see if network can denoise
        # noise_test = torch.randn(1, 1, 128, 109).to('cuda') # batch size x channel x flattened latent size
        # pandas = torch.tensor(np.load("data/panda/pandas.npy")).to('cuda')
        # pandas = pandas / pandas.std()
        # pandas = negOneToOneNorm(pandas)
        # x_t = torch.sqrt(gamma(torch.tensor(0.01))) * SCALE * pandas + torch.sqrt(1-gamma(torch.tensor(0.99))) * noise_test


        x_t_next = x_t
        start_noise  = x_t
        with torch.no_grad():
            #code below actually performs the sampling
            for n in tqdm(reversed(range(TIME_STEPS))):
                x_t = x_t_next
                t_now = torch.tensor(n).to(x_t.device).unsqueeze(0)
                t_next = t_now - 1

                signal_rate_now = torch.zeros_like(t_now).float()
                for i in range(t_now.shape[0]):
                    signal_rate_now[i] = gamma(t_now[i] / TIME_STEPS).item()
                signal_rate_now = signal_rate_now.unsqueeze(1).unsqueeze(1)
                noise_rate_now = 1 - signal_rate_now

                signal_rate_next = torch.zeros_like(t_next).float()
                for i in range(t_next.shape[0]):
                    signal_rate_next[i] = gamma(t_next[i] / TIME_STEPS).item()
                signal_rate_next = signal_rate_next.unsqueeze(1).unsqueeze(1)
                noise_rate_next = 1 - signal_rate_next

                noise_pred = self.forward(x_t, t_now, conditioning_var)
                pred_x_0 = (x_t - torch.sqrt(noise_rate_now) * noise_pred) / (torch.sqrt(signal_rate_now) * SCALE)
                
                if n > 0:
                    if n % 10 == 0:
                        #print mse loss pred_x_0 and pandas
                        if target_sample_for_comparison is not None:
                            print(f"mse loss pred_x_0 and target sample at step: {n}: ", F.mse_loss(pred_x_0.squeeze(0).squeeze(0), target_sample_for_comparison))

                        #print signal rates
                        print("signal_rate_now: ", signal_rate_now)
                        print("signal_rate_next: ", signal_rate_next)
                        print(f"MSE between noise and pred at timestep t: {n}: ", F.mse_loss(noise_pred, start_noise))
                    signal_rate_next = gamma(t_next / TIME_STEPS)
                    noise_rate_next = 1 - signal_rate_next
                    np.save("output/samples/every_sample_step/every_sample_step_" + str(n), pred_x_0.squeeze(0).squeeze(0).reshape(128, 109).cpu().numpy().clip(-1, 1), -1.0, 1.0)
                    noise = torch.randn(x_t.shape).to(x_t.device)
                    if SAMPLING_ALGORITHM == 'DDIM':
                        # eta = 0
                        # c1 = eta * ((1 - signal_rate_now / signal_rate_next) * (1 - signal_rate_next) / (1 - signal_rate_now)).sqrt()
                        # c2 = torch.sqrt((noise_rate_next - c1 ** 2))
                        # x_t_next = torch.sqrt(signal_rate_next) * SCALE * pred_x_0 + c2 * noise_pred + c1 * noise
                        x_t_next = torch.sqrt(signal_rate_next) * SCALE * pred_x_0 + torch.sqrt(noise_rate_next) * noise
                    else:
                        x_t_next = torch.sqrt(signal_rate_next) * SCALE * pred_x_0 + torch.sqrt(noise_rate_next) * noise


        #clip to -1, 1
        # x_t = torch.clamp(x_t, -1.0, 1.0)
        return pred_x_0.squeeze(1)
    
    #generate a sample from noise input
    def sample_xt(self, x_t, target_sample_for_comparison, conditioning_var=None):
        x_t_next = x_t
        with torch.no_grad():
            #sample t-1 sample directly, do not predict noise
            for n in tqdm(reversed(range(TIME_STEPS))):
                x_t = x_t_next
                t_now = torch.tensor(n).unsqueeze(0).to(x_t.device)
                t_next = t_now - 1

                signal_rate_now = torch.zeros_like(t_now).float()
                for i in range(t_now.shape[0]):
                    signal_rate_now[i] = gamma(t_now[i] / TIME_STEPS).item()
                signal_rate_now = signal_rate_now.unsqueeze(1).unsqueeze(1)
                noise_rate_now = 1 - signal_rate_now

                signal_rate_next = torch.zeros_like(t_next).float()
                for i in range(t_next.shape[0]):
                    signal_rate_next[i] = gamma(t_next[i] / TIME_STEPS).item()
                signal_rate_next = signal_rate_next.unsqueeze(1).unsqueeze(1)
                noise_rate_next = 1 - signal_rate_next

                pred_x_0 = self.forward(x_t, t_now, conditioning_var)
                np.save("output/samples/every_sample_step/every_sample_step_" + str(n), x_t.squeeze(0).squeeze(0).reshape(128, 109).cpu().numpy())

                if n > 0:
                    noise = torch.randn(x_t.shape).to(x_t.device)
                    x_t_next = torch.sqrt(signal_rate_next) * SCALE * pred_x_0 + torch.sqrt(noise_rate_next) * noise

        return pred_x_0.squeeze(1)


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_per_diffusion_step = np.zeros((10,))
    def training_step(self, batch, batch_idx):

        device = self.device
        conditioning_var = None
        if WITH_CONDITIONING:
            x_0 = batch[0] # batch size, channels, length 
            conditioning_var = batch[1]
        else:
            x_0 = batch
        t = torch.randint(1, TIME_STEPS, (x_0.shape[0],)).to(device)
        noise = torch.randn(x_0.shape).to(device)

        signal_rate = torch.zeros_like(t).float()
        for i in range(t.shape[0]):
            signal_rate[i] = gamma(t[i] / TIME_STEPS).item()
        signal_rate = signal_rate.unsqueeze(1).unsqueeze(1)
        noise_rate = 1 - signal_rate
        x_t = torch.sqrt(signal_rate) * SCALE * x_0 + torch.sqrt(noise_rate) * noise

        #TODO:maybe remove later
        #normalize to zero mean and unit variance
        # x_t = (x_t - x_t.mean()) / x_t.std()
        # x_t = negOneToOneNorm(x_t)

        y_pred = self.model.forward(x_t, t, conditioning_var)
        if PRED_NOISE:
            batch_loss = F.mse_loss(y_pred, noise)
        else:
            batch_loss = F.mse_loss(y_pred, x_0)

        self.log('train_loss', batch_loss, on_epoch=True)

        return batch_loss
    
    def validation_step(self, batch, batch_idx):
        device = self.device
        
        conditioning_var = None
        if WITH_CONDITIONING:
            x_0 = batch[0] # batch size, channels, length 
            conditioning_var = batch[1] 
        else:
            x_0 = batch

        t = torch.randint(1, TIME_STEPS, (1,)).to(device)
        noise = torch.randn(x_0.shape).to(device)
        signal_rate = gamma(t / TIME_STEPS)
        noise_rate = 1 - signal_rate
        x_t = torch.sqrt(signal_rate) * SCALE * x_0 + torch.sqrt(noise_rate) * noise

        y_pred = self.model.forward(x_t, t, conditioning_var)
        if PRED_NOISE:
            batch_loss = F.mse_loss(y_pred, noise)
        else:
            batch_loss = F.mse_loss(y_pred, x_0)

        self.log('train_loss_validation', batch_loss, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        return [optimizer], []
    

def simple_linear_schedule(t, clip_min=0.000001):
    # A gamma function that simply is 1-t.
    return torch.tensor(np.clip(1 - t, clip_min, 0.9999))

def cosine_schedule(t, start=0, end=1, tau=1, clip_min=0.000001):
    # A gamma function based on cosine function.
    v_start = math.cos(start * math.pi / 2) ** (2 * tau)
    v_end = math.cos(end * math.pi / 2) ** (2 * tau)
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.tensor(np.clip(output, clip_min, 0.9999))

def exponential_schedule(t, start=0, end=1, tau=1, clip_min=0.000001):
    # A gamma function based on exponential function.
    v_start = math.exp(-start * tau)
    v_end = math.exp(-end * tau)
    output = math.exp(-t * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.tensor(np.clip(output, clip_min, 0.9999))
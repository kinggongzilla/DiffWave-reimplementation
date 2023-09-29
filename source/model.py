import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import wandb
from config import VARIANCE_SCHEDULE, N_MELS, TIME_STEPS, WITH_CONDITIONING, LEARNING_RATE, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, RES_CHANNELS
import datetime

def Conv1d(*args, **kwargs):
  layer = torch.nn.Conv1d(*args, **kwargs)
  torch.nn.init.kaiming_normal_(layer.weight)
  return layer

#Embedding for diffusion time step
class DiffusionEmbedding(torch.nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = torch.nn.Linear(128, 512)
        self.silu1 = torch.nn.SiLU()
        self.projection2 = torch.nn.Linear(512, 512)
        self.silu2 = torch.nn.SiLU()

    #project diffusion timestep into latent space
    def forward(self, t):
        if t.dtype in [torch.int32, torch.int64]:
            x = self.embedding[t]
        else:
            x = self._lerp_embedding(t)
        x = self.projection1(x)
        x = self.silu1(x)
        x = self.projection2(x)
        x = self.silu2(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table

#conditioner for spectrogram
class SpectrogramConditioner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.acivation1 = torch.nn.LeakyReLU(0.4)
        self.conv2 = torch.nn.ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])
        self.acivation2 = torch.nn.LeakyReLU(0.4)

    #project spectrogram into latent space
    def forward(self, spectrogram):
        spectrogram = self.conv1(spectrogram)
        spectrogram = self.acivation1(spectrogram)
        spectrogram = self.conv2(spectrogram)
        spectrogram = self.acivation2(spectrogram)
        return torch.squeeze(spectrogram, 1)[:,:,:SAMPLE_RATE*SAMPLE_LENGTH_SECONDS]

#DiffWave residual block
class DiffWaveBlock(torch.nn.Module):
    def __init__(self, layer_index, residual_channles, layer_width, dilation_mod, with_conditioning: bool) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.residual_channels = residual_channles
        self.layer_width = layer_width
        self.with_conditioner = with_conditioning

        if with_conditioning:
            self.conv_conditioner = Conv1d(N_MELS, 2*residual_channles, 1)

        # linear layer that processes diffusion timestep
        self.fc_timestep = torch.nn.Linear(layer_width, residual_channles)

        # bi directional conv
        self.conv_dilated = Conv1d(residual_channles, 2*residual_channles, 3, dilation=2**(layer_index%dilation_mod), padding='same')

        # outgoing convolution laayer
        self.conv_out = Conv1d(residual_channles, 2*residual_channles, 1)

    #forward pass, according to architecture in DiffWave paper
    def forward(self, x, t, conditioning_var=None):

        t = self.fc_timestep(t)
        t = t.unsqueeze(-1) # add another dimension at the end
        t = t.expand(1, RES_CHANNELS, x.shape[2]) # expand the last dimension to match x;
        
        y = x + t #broadcast addition

        y = self.conv_dilated(y)

        #if conditionin variable is used, add it as bias to input y
        if conditioning_var is not None:
            y = y + self.conv_conditioner(conditioning_var)

        a, b = y.chunk(2, dim=1)

        y = torch.tanh(a) * torch.sigmoid(b)

        y = self.conv_out(y)

        y, skip = torch.chunk(y, 2, dim=1)

        return (y + x) / np.sqrt(2.0), skip


class DiffWave(torch.nn.Module):
    def __init__(self, residual_channels, num_blocks, timesteps, variance_schedule, with_conditioning, n_mels, layer_width=512, dilation_mod=10, ) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.variance_schedule = variance_schedule
        self.num_blocks = num_blocks
        self.layer_width = layer_width
        self.with_conditioner = with_conditioning
        self.n_mels = n_mels

        #check if conditioning is used and set conditioner
        if with_conditioning:
            self.conditioner_block = SpectrogramConditioner()

        #layer that projects diffusion timestep into latent space
        self.timestep_in = DiffusionEmbedding(len(VARIANCE_SCHEDULE))

        #input layer before DiffWave blocks
        self.waveform_in = torch.nn.Sequential(
            Conv1d(1, residual_channels, 1),
            torch.nn.ReLU()
        )

        #DiffWave blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DiffWaveBlock(i, residual_channels, layer_width, dilation_mod=dilation_mod, with_conditioning=with_conditioning))

        #outgoing layers
        self.out = torch.nn.Sequential(
            Conv1d(residual_channels, residual_channels, 1), 
            torch.nn.ReLU(),
            Conv1d(residual_channels, 1, 1))

    #forward pass according to DiffWave paper
    def forward(self, x, t, conditioning_var=None):

        #conditioning variable (spectrogram) input
        if conditioning_var is not None:
            conditioning_var = self.conditioner_block(conditioning_var)

        

        #waveform input
        x = self.waveform_in(x)

        #time embedding
        t = self.timestep_in(t)

        #blocks
        skip = None
        for block in self.blocks:
            x, skip_connection = block.forward(x, t, conditioning_var=conditioning_var)
            skip = skip_connection if skip is None else skip_connection + skip
        skip = skip / np.sqrt(len(self.blocks)) #divide by sqrt of number of blocks as in paper Github code

        #out
        x = self.out(x)
        
        return x

      #generate a sample from noise input
    def sample(self, x_t, conditioning_var=None):
        with torch.no_grad():

        #sample t-1 sample directly, do not predict noise
            for n in tqdm(range(len(VARIANCE_SCHEDULE) - 2, -1, -1)):
                x_t = self.forward(x_t, torch.tensor(n+1), conditioning_var)

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

        noise = noise.to(device)

        noisy_latent = (1- beta[t])*latent + (beta[t])*noise
        less_noisy_latent = (1- beta[t-1])*latent + (beta[t-1])*noise

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
    

class TimeLogger():
    def __init__(self, message):
        self.message = message
        self.start_time = None
        self.end_time = None
        self.duration = None

    def start(self):
        self.start_time = datetime.datetime.now()

    def stop(self):
        self.end_time = datetime.datetime.now()
        self.duration = self.end_time - self.start_time
        print("Time used by step ", self.message, ": ", self.duration)
        return self.duration
    

    
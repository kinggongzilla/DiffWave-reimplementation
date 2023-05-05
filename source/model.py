import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
import wandb
from config import VARIANCE_SCHEDULE, N_MELS, TIME_STEPS, WITH_CONDITIONING, LEARNING_RATE, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS

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
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = torch.nn.Linear(512, residual_channels)
        if not uncond: # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
            (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / np.sqrt(2.0), skip
    
    


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
            self.blocks.append(DiffWaveBlock(n_mels, residual_channels, dilation=2**(i % dilation_mod), uncond=False))

        #outgoing layers
        self.out = torch.nn.Sequential(
            Conv1d(residual_channels, residual_channels, 1), 
            torch.nn.ReLU(),
            Conv1d(residual_channels, 1, 1))

    #forward pass according to DiffWave paper
    def forward(self, x, t, conditioning_var=None):
        print('beginning forward Diffwave:')
        print(torch.cuda.memory_allocated())
        #conditioning variable (spectrogram) input
        if conditioning_var is not None:
            conditioning_var = self.conditioner_block(conditioning_var)
        
        print('after conditioning block in forward Diffwave:')
        print(torch.cuda.memory_allocated())

        #waveform input
        x = self.waveform_in(x)

        print('after waveform_in in forward Diffwave:')
        print(torch.cuda.memory_allocated())

        #time embedding
        t = self.timestep_in(t)

        print('after timestep_in  in forward Diffwave:')
        print(torch.cuda.memory_allocated())

        #blocks
        skip = None
        for block in self.blocks:
            x, skip_connection = block.forward(x, t, conditioner=conditioning_var)
            skip = skip_connection if skip is None else skip_connection + skip
        skip = skip / np.sqrt(len(self.blocks)) #divide by sqrt of number of blocks as in paper Github code
        
        print('after blocks for loop in forward Diffwave:')
        print(torch.cuda.memory_allocated())

        #out
        x = self.out(x)
        print('forward DiffWave memory allocated: ')
        print(torch.cuda.memory_allocated())
        return x

    #generate a sample from noise input
    def sample(self, x_t, conditioning_var=None):
        with torch.no_grad():

            beta = self.variance_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            #code below actually performs the sampling
            for n in tqdm(range(len(alpha) - 1, -1, -1)):
                c1 = 1 / alpha[n]**0.5 # c1 approaches 1 as timestep gets closer to 0
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                x_t = c1 * (x_t - c2 * self.forward(x_t, torch.tensor(n), conditioning_var).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(x_t)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    x_t += sigma * noise
                x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t 


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        print('after init model memory allocated: ')
        print(torch.cuda.memory_allocated())
    def training_step(self, batch, batch_idx):
        print('beginning training_step memory allocated: ')
        print(torch.cuda.memory_allocated())
        #get waveform from (waveform, sample_rate) tuple;
        waveform = batch[0] # batch size, channels, length 

        #generate noise
        noise = torch.randn(waveform.shape)

        #generate random integer between 1 and number of diffusion timesteps
        t = torch.randint(0, TIME_STEPS, (1,))

        #define scaling factors for original waveform and noise

        beta = VARIANCE_SCHEDULE
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        print('before noise.to(device) memory allocated: ')
        print(torch.cuda.memory_allocated())


        #put all tensors on correct device
        device = self.device
        alpha_cum = alpha_cum.to(device)
        noise = noise.to(device)

        print('after noise.to(device) memory allocated: ')
        print(torch.cuda.memory_allocated())

        #create noisy version of original waveform
        waveform = torch.sqrt(alpha_cum[t])*waveform + torch.sqrt(1-alpha_cum[t])*noise

        print('after noisy waveform creation: ')
        print(torch.cuda.memory_allocated())

        del alpha_cum, beta, alpha

        conditioning_var = None
        if WITH_CONDITIONING:
            # get conditioning_var (spectrogram) from (waveform, sample_rate, spectrogram) tuple;
            conditioning_var = batch[2] # batch size, channels, length

        print('after assigning conditioning_var:')
        print(torch.cuda.memory_allocated())

        # predict noise at diffusion timestep t
        y_pred = self.model.forward(waveform, t, conditioning_var)

        del t

        #calculate loss and return loss
        batch_loss = F.l1_loss(y_pred, noise)

        del noise

        self.log('train_loss', batch_loss, on_epoch=True)

        print('end training_step memory allocated: ')
        print(torch.cuda.memory_allocated())

        return batch_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        return [optimizer], []
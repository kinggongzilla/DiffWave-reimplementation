import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
import numpy as np
from config import VARIANCE_SCHEDULE, N_MELS, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS

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
        # self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 7), padding=(1, 128)) #tanspose conv shapes for speech samples
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 7), padding=(1, 128))
        self.acivation1 = torch.nn.LeakyReLU(0.4)
        # self.conv2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 15), padding=(1, 229), output_padding=(0, 1)) #transpose conv shapes for speech samples
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 7), padding=(1, 29))
        self.acivation2 = torch.nn.LeakyReLU(0.4)

    #project spectrogram into latent space
    def forward(self, spectrogram):
        spectrogram = self.conv1(spectrogram)
        spectrogram = self.acivation1(spectrogram)
        spectrogram = self.conv2(spectrogram)
        spectrogram = self.acivation2(spectrogram)
        return torch.squeeze(spectrogram, 1)

#DiffWave residual block
class DiffWaveBlock(torch.nn.Module):
    def __init__(self, layer_index, residual_channles, layer_width, dilation_mod, with_conditioning: bool) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.residual_channels = residual_channles
        self.layer_width = layer_width
        self.input = None
        self.x_skip = None
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
        input = x.clone()
        t = self.fc_timestep(t)
        t = t.unsqueeze(-1) # add another dimension at the end
        t = t.expand(1, 64, SAMPLE_RATE * SAMPLE_LENGTH_SECONDS) # expand the last dimension to match x; 22500 * 5 = 110250
        x = x + t #broadcast addition
        x = self.conv_dilated(x)

        #if conditionin variable is used, add it as bias to input x
        if conditioning_var is not None:
            x = x + self.conv_conditioner(conditioning_var)
        x_tanh, x_sigmoid = x.chunk(2, dim=1)
        x_tanh = torch.tanh(x_tanh)
        x_sigmoid = torch.sigmoid(x_sigmoid)
        x = x_tanh * x_sigmoid
        x = self.conv_out(x)
        x, skip = torch.chunk(x, 2, dim=1)
        return (x + input) / np.sqrt(2.0), skip


class DiffWave(torch.nn.Module):
    def __init__(self, residual_channels, num_blocks, timesteps, variance_schedule, with_conditioning, n_mels, layer_width=512, dilation_mod=12, ) -> None:
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

            talpha = 1 - self.variance_schedule
            talpha_cum = np.cumprod(talpha)

            beta = self.variance_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            #code below calculates the time steps for the diffusion process; relevant for FAST sampling
            T = []
            for s in range(len(self.variance_schedule)):
                for t in range(len(self.variance_schedule) - 1):
                    if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            #the code below is the actual sampling process
            for n in tqdm(range(len(alpha) - 1, -1, -1)):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                x_t = c1 * (x_t - c2 * self.forward(x_t, torch.tensor(n), conditioning_var).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(x_t)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    x_t += sigma * noise
                x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t 



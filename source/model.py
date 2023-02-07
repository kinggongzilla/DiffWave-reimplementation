import math
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
import numpy as np

class DiffWaveBlock(torch.nn.Module):
    def __init__(self, layer_index, residual_channles, layer_width, dilation_mod) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.residual_channels = residual_channles
        self.layer_width = layer_width
        self.input = None
        self.x_skip = None

        # diffusion time step embedding
        self.fc_timestep = torch.nn.Linear(layer_width, residual_channles)

        #bi directional conv
        self.conv_dilated = torch.nn.Conv1d(residual_channles, 2*residual_channles, 3, dilation=2**(layer_index%dilation_mod), padding='same')

        self.conv_skip = torch.nn.Conv1d(residual_channles, residual_channles, 1)
        self.conv_next = torch.nn.Conv1d(residual_channles, residual_channles, 1)

    def forward(self, x, t):

        input = x.clone()
        t = self.fc_timestep(t)
        t = torch.broadcast_to(torch.unsqueeze(t,2), (x.shape[0], x.shape[1], x.shape[2])) #broadcast to length of audio input
        x = x + t #broadcast addition
        x = self.conv_dilated(x)
        x_tanh, x_sigmoid = x.chunk(2, dim=1)
        x_tanh = torch.tanh(x_tanh)
        x_sigmoid = torch.sigmoid(x_sigmoid)
        x = x_tanh * x_sigmoid
        self.x_skip = self.conv_skip(x)
        x = self.conv_next(x) + input
        return x


class DiffWave(torch.nn.Module):
    def __init__(self, residual_channels, num_blocks, timesteps, variance_schedule, layer_width=512, dilation_mod=12) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.variance_schedule = variance_schedule
        self.num_blocks = num_blocks
        self.layer_width = layer_width

        #in
        self.timestep_in = torch.nn.Sequential(
            torch.nn.Linear(128, layer_width),
            torch.nn.SiLU(),
            torch.nn.Linear(layer_width, layer_width),
            torch.nn.SiLU()
        )

        self.waveform_in = torch.nn.Sequential(
            torch.nn.Conv1d(1, residual_channels, 1),
            torch.nn.ReLU()
        )

        #blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(DiffWaveBlock(i, residual_channels, layer_width, dilation_mod=dilation_mod))

        #out
        self.out = torch.nn.Sequential(
            torch.nn.Conv1d(residual_channels, residual_channels, 1), 
            torch.nn.Conv1d(residual_channels, 1, 1))

    def forward(self, x, t):

        #waveform input
        x = self.waveform_in(x)

        #time embedding
        t = self.embed_timestep(t).to(x.device)

        t = self.timestep_in(t)

        residual_sum = torch.empty_like(x)

        #blocks
        for block in self.blocks:
            x = block(x, t)
            residual_sum += block.x_skip

        #out
        x = self.out(x)
        return x


    def sample(self, x_t):
        with torch.no_grad():

            talpha = 1 - self.variance_schedule
            talpha_cum = np.cumprod(talpha)

            beta = self.variance_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(self.variance_schedule)):
                for t in range(len(self.variance_schedule) - 1):
                    if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)


            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                x_t = c1 * (x_t - c2 * self.forward(x_t, n).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(x_t)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    x_t += sigma * noise
                x_t = torch.clamp(x_t, -1.0, 1.0)
        return x_t 
            

    def embed_timestep(self, t, batch_size=1):
        embedding = torch.zeros(1, 128)
        for i in range(64):
            embedding[0, i] = math.sin(10**((i*4)/63)*t)
        for j in range(64):
            embedding[0, j+64] = math.cos(10**((j*4)/63)*t)
        return torch.broadcast_to(embedding, (batch_size, 128)) #broadcast to batch size



import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchaudio
import numpy as np

class SpectrogramConditioner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 7), padding=(1, 128)) #tanspose conv shapes for speech samples
        self.conv1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 7), padding=(1, 128))
        self.acivation1 = torch.nn.LeakyReLU(0.4)
        # self.conv2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 15), padding=(1, 229), output_padding=(0, 1)) #transpose conv shapes for speech samples
        self.conv2 = torch.nn.ConvTranspose2d(1, 1, kernel_size=(3,12), stride=(1, 7), padding=(1, 29))
        self.acivation2 = torch.nn.LeakyReLU(0.4)

    def forward(self, spectrogram):
        spectrogram = self.conv1(spectrogram)
        spectrogram = self.acivation1(spectrogram)
        spectrogram = self.conv2(spectrogram)
        spectrogram = self.acivation2(spectrogram)
        return torch.squeeze(spectrogram, 1)


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
            self.conv_conditioner = torch.nn.Conv1d(80, 2*residual_channles, 1)

        # diffusion time step embedding
        self.fc_timestep = torch.nn.Linear(layer_width, residual_channles)

        #bi directional conv
        self.conv_dilated = torch.nn.Conv1d(residual_channles, 2*residual_channles, 3, dilation=2**(layer_index%dilation_mod), padding='same')

        self.conv_out = torch.nn.Conv1d(residual_channles, 2*residual_channles, 1)
        # self.conv_skip = torch.nn.Conv1d(residual_channles, residual_channles, 1)
        # self.conv_next = torch.nn.Conv1d(residual_channles, residual_channles, 1)

    def forward(self, x, t, conditioning_var=None):
        input = x.clone()
        t = self.fc_timestep(t)
        t = torch.broadcast_to(torch.unsqueeze(t,2), (x.shape[0], x.shape[1], x.shape[2])) #broadcast to length of audio input
        x = x + t #broadcast addition
        x = self.conv_dilated(x)
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

        #conditioning_var
        if with_conditioning:
            self.conditioner_block = SpectrogramConditioner()

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
            self.blocks.append(DiffWaveBlock(i, residual_channels, layer_width, dilation_mod=dilation_mod, with_conditioning=with_conditioning))

        #out
        self.out = torch.nn.Sequential(
            torch.nn.Conv1d(residual_channels, residual_channels, 1), 
            torch.nn.ReLU(),
            torch.nn.Conv1d(residual_channels, 1, 1))

    def forward(self, x, t, conditioning_var=None):
        #conditioning variable (spectrogram) input
        if conditioning_var is not None:
            conditioning_var = self.conditioner_block(conditioning_var)

        #waveform input
        x = self.waveform_in(x)

        #time embedding
        t = self.embed_timestep(t).to(x.device)

        t = self.timestep_in(t)

        residual_sum = torch.empty_like(x)

        #blocks
        skip = None
        for block in self.blocks:
            x, skip_connection = block.forward(x, t, conditioning_var=conditioning_var)
            skip = skip_connection if skip is None else skip_connection + skip
        skip = skip / np.sqrt(len(self.blocks)) #divide by sqrt of number of blocks as in paper Github code
        #out
        x = self.out(skip)
        return x


    def sample(self, x_t, conditioning_var=None):
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


            for n in tqdm(range(len(alpha) - 1, -1, -1)):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                x_t = c1 * (x_t - c2 * self.forward(x_t, n, conditioning_var).squeeze(1))
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



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

#Takes latent sample as input
# transpose convolutional block that takes 1 channel of shape (128, 109) as input and outputs 1 channel of shape (128, 128)
class input_latent(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_c, 16, kernel_size=(3,9), padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 32, kernel_size=(3,9), padding=1)
        self.conv3 = nn.ConvTranspose2d(32, out_c, kernel_size=(3,10), padding=1)
        self.tanh = nn.Tanh()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x
    
class input_spectrogram(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 16, kernel_size=(3,3), stride=(1,1), padding=(1,11))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(1,1), padding=(1,11))
        self.conv3 = nn.Conv2d(32, out_c, kernel_size=(3,3), stride=(1,1), padding=(1,12))
        self.pool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=(0,1))
        self.tanh = nn.Tanh()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.tanh(x)
        return x


#try interpolation instead of convoultions 
class SpectrogramDownscaler(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, input):
        # input is a tensor of shape (batch_size, channels, height, width)
        # output_size is a tuple of (height, width)
        output = F.interpolate(input, size=self.output_size, mode="bilinear", align_corners=False) # try different modes: “nearest”, “bicubic”, or “area”
        return output
    

#output block

class output_latent(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=(3,8), stride=(1,1), padding=(1,0))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3,7), stride=(1,1), padding=(1,0))
        self.conv4 = nn.Conv2d(16, out_c, kernel_size=(3,7), stride=(1,1), padding=(1,0))
        self.tanh = nn.Tanh()
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.tanh(x)
        x = self.conv2(x)
        x = self.tanh(x)
        x = self.conv3(x)
        x = self.tanh(x)
        x = self.conv4(x)
        # x = self.tanh(x)
        return x
    
class input_timestep(torch.nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = torch.nn.Linear(128, 128)
        self.silu1 = torch.nn.SiLU()
        self.projection2 = torch.nn.Linear(128, 128)
        self.silu2 = torch.nn.SiLU()


    #project diffusion timestep into latent space
    def forward(self, t):
        x = self.create_timestep_embedding(t, 128)
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
    
    def create_timestep_embedding(self, t: float, dim: int):
        assert 0 <= t <= 1, "Input t must be a floating point number between 0 and 1"

        # Create a tensor of size `dim` for the timestep embedding
        timestep_embedding_cos = torch.zeros(dim)
        timestep_embedding_sin = torch.zeros(dim)
        
        # Fill the tensor with values derived from the input `t`
        for i in range(dim):
            timestep_embedding_cos[i] = t * math.cos(i / dim * math.pi * t)
            timestep_embedding_sin[i] = t * math.sin(i / dim * math.pi * (1-t))
        
        # Concatenate the tensors together and return
        return (timestep_embedding_cos + timestep_embedding_sin).to('cuda')
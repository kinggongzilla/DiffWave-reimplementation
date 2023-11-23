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
class output_layer(nn.Module):
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
        return x
    
class input_timestep(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.projection1 = torch.nn.Linear(128, 128)
        self.silu1 = torch.nn.SiLU()
        self.projection2 = torch.nn.Linear(128, 128)
        self.silu2 = torch.nn.SiLU()
        self.upsample = torch.nn.Upsample(size=(128, 128), mode='nearest')

    #project diffusion timestep into latent space
    def forward(self, t):
        #apply create_time_step_embedding to t for every sample in batch
        embeddings = torch.zeros((t.shape[0], 128)).to('cuda')
        for i in range(t.shape[0]):
            embeddings[i] = self.create_timestep_embedding(t[i], 128)
        embeddings = self.projection1(embeddings)
        embeddings = self.silu1(embeddings)
        embeddings = self.projection2(embeddings)
        embeddings = self.silu2(embeddings)
        embeddings = torch.unsqueeze(embeddings, 1)
        embeddings = torch.unsqueeze(embeddings, 1)
        embeddings = self.upsample(embeddings)
        return embeddings

    def create_timestep_embedding(self, t: float, dim: int):
        embedding_min_frequency = 1.0
        embedding_dims = 128
        embedding_max_frequency = 1000.0
        frequencies = torch.exp(
            torch.linspace(
                math.log(embedding_min_frequency),
                math.log(embedding_max_frequency),
                embedding_dims // 2,
            )
        ).to('cuda')
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = torch.cat(
            [torch.sin(angular_speeds * t), torch.cos(angular_speeds * t)], 
        )
        return embeddings
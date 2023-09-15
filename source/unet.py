import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class conv_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_c)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_c)
    self.relu = nn.ReLU()
  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    return x
  
class encoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv = conv_block(in_c, out_c)
    self.pool = nn.MaxPool2d((2, 2))
  def forward(self, inputs):
    x = self.conv(inputs)
    p = self.pool(x)
    return x, p

class decoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
    self.conv = conv_block(out_c+out_c, out_c)
  def forward(self, inputs, skip):
    x = self.up(inputs)
    x = torch.cat([x, skip], axis=1)
    x = self.conv(x)
    return x

class UNet(nn.Module):
  def __init__(self, in_c=3, out_c=1):
    super().__init__()
    """ Encoder """
    self.e1 = encoder_block(in_c, 64)
    self.e2 = encoder_block(64, 128)
    self.e3 = encoder_block(128, 256)
    self.e4 = encoder_block(256, 512)
    """ Bottleneck """
    self.b = conv_block(512, 1024)
    """ Decoder """
    self.d1 = decoder_block(1024, 512)
    self.d2 = decoder_block(512, 256)
    self.d3 = decoder_block(256, 128)
    self.d4 = decoder_block(128, 64)
    """ Classifier """
    self.outputs = nn.Conv2d(64, out_c, kernel_size=1, padding=0)

  def forward(self, inputs, t, conditioning_var=None):
    """ Encoder """
    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)
    """ Bottleneck """
    b = self.b(p4)
    """ Decoder """
    d1 = self.d1(b, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)
    """ Classifier """
    outputs = self.outputs(d4)
    return outputs

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
          # sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
          sigma = beta[n]**0.5 #this sigma is optimal if x_0 is from gaussian; see section 3.2 Denoising Diffusion Probabilistic Models
          x_t += sigma * noise

          #print c1, c2 and sigma every 100 timesteps
          if n % 100 == 0:
            print("c1: ", c1, "c2: ", c2, "sigma: ", sigma)


#UNet Code from: https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
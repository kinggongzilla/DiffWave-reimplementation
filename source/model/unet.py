import torch
import torch.nn as nn
from config import WITH_DROPOUT

class conv_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
    self.bn1 = nn.BatchNorm2d(out_c)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
    self.bn2 = nn.BatchNorm2d(out_c)
    self.relu = nn.ReLU()
    if WITH_DROPOUT:
      self.dropout = nn.Dropout2d(0.1)

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    if WITH_DROPOUT:
      x = self.dropout(x) # apply dropout after activation
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
    self.e1 = encoder_block(in_c, int(64 * 2))
    self.e2 = encoder_block(int(64 * 2), int(128 * 2))
    self.e3 = encoder_block(int(128 * 2), int(256 * 2))
    self.e4 = encoder_block(int(256 * 2), int(512 * 2))
    """ Bottleneck """
    self.b = conv_block(int(512 * 2), int(1024 * 2))
    """ Decoder """
    self.d1 = decoder_block(int(1024 * 2), int(512 * 2))
    self.d2 = decoder_block(int(512 * 2), int(256 * 2))
    self.d3 = decoder_block(int(256 * 2), int(128 * 2))
    self.d4 = decoder_block(int(128 * 2), int(64 * 2))
    """ Classifier """
    self.outputs = nn.Conv2d(int(64 * 2), out_c, kernel_size=1, padding=0)

  def forward(self, inputs,):
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
    """ Output """
    outputs = self.outputs(d4)
    return outputs





#UNet Code from: https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
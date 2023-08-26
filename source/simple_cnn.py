import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define three convolutional layers with kernel size 3, stride 1 and padding 1
        # The first layer has 1 input channel and 16 output channels
        # The second layer has 16 input channels and 32 output channels
        # The third layer has 32 input channels and 1 output channel
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, x, t, conditioning_var=None):

        #unsqueeze x at first dimension
        x = x.unsqueeze(1)

        # Apply the first convolutional layer and a ReLU activation function
        x = F.relu(self.conv1(x))
        # Apply the second convolutional layer and a ReLU activation function
        x = F.relu(self.conv2(x))
        # Apply the third convolutional layer and a sigmoid activation function
        x = torch.sigmoid(self.conv3(x))
        # Return the output
        return x
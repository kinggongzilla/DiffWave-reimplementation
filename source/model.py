import math
import torch
import torch.nn.functional as F
import torchaudio

class DiffWaveBlock(torch.nn.Module):
    def __init__(self, layer_index, C) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.C = C
        self.input = None
        self.x_skip = None

        # diffusion time step embedding
        self.fc_timestep = torch.nn.Linear(512, C)

        #bi directional conv
        self.conv_dilated = torch.nn.Conv1d(C, 2*C, 3, dilation=2**layer_index, padding='same')

        self.conv_skip = torch.nn.Conv1d(C, C, 1)
        self.conv_next = torch.nn.Conv1d(C, C, 1)

    def forward(self, x, t):

        self.input = x.clone()
        t = self.fc_timestep(t)
        t = torch.broadcast_to(torch.unsqueeze(t,2), (x.shape[0], x.shape[1], x.shape[2])) #broadcast to length of audio input
        x = x + t #broadcast addition
        x = self.conv_dilated(x)
        x_tanh, x_sigmoid = x.chunk(2, dim=1)
        x_tanh = torch.tanh(x_tanh)
        x_sigmoid = torch.sigmoid(x_sigmoid)
        x = x_tanh * x_sigmoid
        self.x_skip = self.conv_skip(x)
        x = self.conv_next(x) + self.input
        return x


class DiffWave(torch.nn.Module):
    def __init__(self, C, timesteps, variance_schedule) -> None:
        super().__init__()
        self.timesteps = timesteps
        self.variance_schedule = variance_schedule

        #in
        self.fc1 = torch.nn.Linear(128, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.conv_in_1 = torch.nn.Conv1d(1, C, 1)

        #blocks
        self.layer1 = DiffWaveBlock(0, C)
        self.layer2 = DiffWaveBlock(1, C)

        #out
        self.conv_out_1 = torch.nn.Conv1d(C, C, 1)
        self.conv_out_2 = torch.nn.Conv1d(C, 1, 1)

    def forward(self, x, t):

        #waveform input
        x = self.conv_in_1(x)

        #time embedding
        t=self.embed_timestep(t)
        t = self.fc1(t)
        t = F.silu(t)
        t = self.fc2(t)
        t = F.silu(t)

        #blocks
        x = self.layer1(x, t)
        x = self.layer2(x, t)

        #out
        x = self.conv_out_1(x)
        x = self.conv_out_2(x)
        return x


    def sample(self, x_t):
        for t in reversed(range(1, self.timesteps)):
            y_pred = self.forward(x_t, t)
            beta = self.variance_schedule[t]
            alpha = 1-beta
            alpha_t = alpha**t
            beta_t = (1-(alpha_t/alpha))/(1-alpha_t) * beta

            mu = 1/torch.sqrt(alpha) * (x_t - (beta/torch.sqrt(1-alpha_t)*y_pred))
            sd = torch.sqrt(beta_t)
            # print('beta: ', beta)
            # print('alpha: ', alpha)
            # print('alpha_t: ', alpha_t)
            # print('beta_t: ', beta_t)
            # print('(alpha_t/alpha): ', (alpha_t/alpha))
            # print('(1-(alpha_t/alpha))/(1-alpha_t): ', (1-(alpha_t/alpha))/(1-alpha_t))
            # print(torch.eye(mu.shape[0], mu.shape[1]))

            x_t = torch.normal(mu, sd*torch.eye(mu.shape[0], mu.shape[1]))
            waveform = x_t[0].detach()
            path = "./outputs/sample.wav"
            torchaudio.save(path, waveform, 44100)
        return waveform
            

    def embed_timestep(self, t, batch_size=1):
        embedding = torch.zeros(1, 128)
        for i in range(64):
            embedding[0, i] = math.sin(10**((i*4)/63)*t)
        for j in range(64):
            embedding[0, j+64] = math.cos(10**((j*4)/63)*t)
        return torch.broadcast_to(embedding, (batch_size, 128)) #broadcast to batch size



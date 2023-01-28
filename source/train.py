import torch
import tqdm
from source.model import DiffWave


def train(C, num_blocks, trainloader, epochs, timesteps, variance_schedule, lr=1e-4):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiffWave(C, num_blocks, timesteps, variance_schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    model.to(device)

    for epoch in range(epochs):
        for i, x in enumerate(trainloader):
            x = x[0] #get waveform from tuple; batch size, channels, length

            optimizer.zero_grad()

            #generate noise and noisy input for model Algorithm 1 Diffwave paper
            noise = torch.randn(x.shape)
            t = torch.randint(1, timesteps, (1,))
            beta = variance_schedule[t]
            alpha = 1-beta
            alpha_t = alpha**t
            
            x = torch.sqrt(alpha_t)*x + torch.sqrt(1-alpha_t)*noise
            x.to(device)
            t.to(device)

            noise = noise.to(device)
            y_pred = model.forward(x, t)
            loss_func = torch.nn.MSELoss(reduction='mean')
            loss = loss_func(y_pred, noise)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'epoch: {epoch} | batch: {i} | loss: {loss.item()}')
        print(f'Epoch {epoch} DONE!')

    torch.save(model.state_dict(), 'model.pt')
    return model
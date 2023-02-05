import torch
from tqdm import tqdm
from source.model import DiffWave
import wandb
from source.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS

wandb.init(
    project="DiffWave", 
    entity="daavidhauser",
    config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_blocks": NUM_BLOCKS,
    "res_channels": RES_CHANNELS,
    "time_steps": TIME_STEPS,
    "variance_schedule": VARIANCE_SCHEDULE,
    "layer_width": LAYER_WIDTH,
    "sample_rate": SAMPLE_RATE,
    "sample_length_seconds": SAMPLE_LENGTH_SECONDS,
    }
)

def train(model, optimizer, trainloader, epochs, timesteps, variance_schedule, lr=1e-4):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    print(f'Total number of parameters: {sum(param.numel() for param in model.parameters())}') #print number of parameters

    model.train()
    model.to(device)
    best_loss = 999999999999
    for epoch in range(epochs):
        epoch_loss = 0
        for x in tqdm(trainloader):
            optimizer.zero_grad()

            x = x[0] #get waveform from tuple; batch size, channels, length            
            noise = torch.randn(x.shape) #generate noise and noisy input for model Algorithm 1 Diffwave paper
            t = torch.randint(1, timesteps, (1,))

            beta = variance_schedule[t]
            alpha = 1-beta
            alpha_t = alpha**t
            
            x = torch.sqrt(alpha_t)*x + torch.sqrt(1-alpha_t)*noise

            x = x.to(device)
            t = t.to(device)
            noise = noise.to(device)

            y_pred = model.forward(x, t)
            del x
            del t
            loss_func = torch.nn.MSELoss(reduction='mean')
            batch_loss = loss_func(y_pred, noise)
            del y_pred
            del noise
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item())
            wandb.log({"batch_loss": batch_loss})
        epoch_loss = epoch_loss/len(trainloader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'output/models/best_model.pt')
        print(f'epoch: {epoch} | loss: {epoch_loss}')
        wandb.log({"epoch_loss": epoch_loss})


    torch.save(model.state_dict(), 'output/models/model.pt')
    return model
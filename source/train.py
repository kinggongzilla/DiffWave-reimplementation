import torch
import numpy as np
from tqdm import tqdm
from source.model import DiffWave
import wandb
from source.config import EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, TIMESTEP_LAYER_WIDTH, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, WITH_CONDITIONING

def train(model, optimizer, trainloader, epochs, timesteps, variance_schedule, lr=1e-4, with_conditioning=WITH_CONDITIONING):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Using device: {device}')
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Total number of parameters: {params}') #print number of parameters

    model.train()
    model.to(device)
    best_loss = 999999999999

    step_count = 0
    step_loss = 0
    best_step_loss = 999999999999
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(trainloader):
            step_count += 1
            optimizer.zero_grad()

            waveform = batch[0] #get waveform from tuple; batch size, channels, length 
            noise = torch.randn(waveform.shape) #generate noise and noisy input for model Algorithm 1 Diffwave paper
            t = torch.randint(1, timesteps, (1,))

            beta = variance_schedule[t]
            alpha = 1-beta
            alpha_t = alpha**t
            
            waveform = torch.sqrt(alpha_t)*waveform + torch.sqrt(1-alpha_t)*noise

            waveform = waveform.to(device)
            t = t.to(device)
            noise = noise.to(device)

            conditioning_var = None
            if with_conditioning:
                conditioning_var = batch[2] # get conditioning_var (spectrogram) from tuple; batch size, channels, length
                conditioning_var = conditioning_var.to(device)

            y_pred = model.forward(waveform, t, conditioning_var)
            del waveform
            del t
            loss_func = torch.nn.MSELoss(reduction='mean')
            batch_loss = loss_func(y_pred, noise)
            del y_pred
            del noise
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item())
            step_loss += float(batch_loss.item())
            wandb.log({"batch_loss": batch_loss})

            #step loss is logged and model saved every 500 steps, so runs with batch size 1 and many, many epochs can be monitored better
            if step_count % 500 == 0:
                wandb.log({"500_step_loss": step_loss/500})
                if best_step_loss > step_loss/500:
                    best_step_loss = step_loss/500
                    torch.save(model.state_dict(), 'output/models/best_500_step_model.pt')
                step_loss = 0

        epoch_loss = epoch_loss/len(trainloader)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'output/models/best_model.pt')
        print(f'epoch: {epoch} | loss: {epoch_loss}')
        wandb.log({"epoch_loss": epoch_loss})

    torch.save(model.state_dict(), 'output/models/model.pt')
    wandb.save('output/models/best_model.pt')
    return model
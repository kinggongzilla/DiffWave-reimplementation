import torch
import numpy as np
from tqdm import tqdm
from model import DiffWave
import wandb
from config import WITH_CONDITIONING

def train(model, optimizer, trainloader, epochs, timesteps, variance_schedule, lr=1e-4, with_conditioning=WITH_CONDITIONING):

    #check if cuda is availableand set as device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #get and print number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Total number of parameters: {params}') #print number of parameters


    model.train()
    model.to(device)

    loss_func = torch.nn.MSELoss()

    step_count = 0
    n_step_loss = 0
    best_step_loss = 999999999999
    best_loss = 999999999999
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(trainloader):
            step_count += 1
            optimizer.zero_grad()

            #get waveform from (waveform, sample_rate) tuple;
            waveform = batch[0] # batch size, channels, length 

            #generate noise
            noise = torch.randn(waveform.shape) 

            #generate random integer between 1 and number of diffusion timesteps
            t = torch.randint(1, timesteps, (1,))

            #define scaling factors for original waveform and noise
            beta = variance_schedule
            alpha = 1-beta
            alpha_t = np.cumprod(alpha)

            #create noisy version of original waveform
            waveform = torch.sqrt(alpha_t[t])*waveform + torch.sqrt(1-alpha_t[t])*noise

            waveform = waveform.to(device)
            t = t.to(device)
            noise = noise.to(device)

            conditioning_var = None
            if with_conditioning:
                # get conditioning_var (spectrogram) from (waveform, sample_rate, spectrogram) tuple;
                conditioning_var = batch[2] # batch size, channels, length
                conditioning_var = conditioning_var.to(device)

            # predict noise at diffusion timestep t
            y_pred = model.forward(waveform, t, conditioning_var)
            del waveform
            del t

            #calculate loss, barward pass and optimizer step
            batch_loss = loss_func(y_pred, noise)
            del y_pred
            del noise
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item())
            n_step_loss += float(batch_loss.item())
            wandb.log({"batch_loss": batch_loss})

            #step loss is logged and model saved every 500 steps, so runs with batch size 1 and many, many epochs can be monitored better
            if step_count % 500 == 0:
                wandb.log({"500_step_loss": n_step_loss/500})
                if best_step_loss > n_step_loss/500:
                    best_step_loss = n_step_loss/500
                    torch.save(model.state_dict(), 'output/models/best_500_step_model.pt')
                n_step_loss = 0

        # normalize epoch_loss by total number of samples
        epoch_loss = epoch_loss/len(trainloader)

        #save model if loss is new best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'output/models/best_model.pt')
        print(f'epoch: {epoch} | loss: {epoch_loss}')
        wandb.log({"epoch_loss": epoch_loss})

    #save final model locally
    torch.save(model.state_dict(), 'output/models/last_model.pt')

    #save model with lowest epoch loss to wandb
    wandb.save('output/models/best_model.pt')
    return model
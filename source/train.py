import os
import sys
sys.path.insert(0, '/home/david/JKU/thesis/DiffWave-reimplementation/source')
from model.model import DenoisingModel, LitModel
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
from pytorch_lightning.loggers import WandbLogger
from dataset import LatentsData
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, TIME_STEPS, VARIANCE_SCHEDULE, SAMPLE_RATE, MAX_SAMPLES, WITH_CONDITIONING, PRED_NOISE, WITH_DROPOUT
import datetime

def train(model_output_path='output/models/'):
    #start with empty cache
    torch.cuda.empty_cache()
    torch.manual_seed(42)

    model_output_path = os.path.join(model_output_path, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)

    #set precision
    torch.set_float32_matmul_precision('medium')

    #print device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    #default data location
    # data_path = os.path.join('data/panda')
    # conditional_path = None
    # data_path = os.path.join('data/encoded_audio')
    # conditional_path = os.path.join('data/mel_spectrograms')

    data_path = os.path.join('../data/9ksamples/encoded_audio/')
    conditional_path = os.path.join('../data/9ksamples/mel_spectrograms/') if WITH_CONDITIONING else None
    val_data_path = os.path.join('../data/9ksamples/encoded_audio_validation/')
    val_conditional_path = os.path.join('../data/9ksamples/mel_spectrograms_validation') if WITH_CONDITIONING else None
    model_checkpoint = None

    #example: python main.py path/to/data
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    if len(sys.argv) > 2:
        conditional_path = sys.argv[2]
    if len(sys.argv) > 3:
        model_checkpoint = sys.argv[3]

    #initialize wandb
    wandb_logger = WandbLogger(
        project="DiffWave", 
        entity="daavidhauser",
        name="BATCH_SIZE_" + str(BATCH_SIZE) + "_LEARNING_RATE_" + str(LEARNING_RATE) + "_TIME_STEPS_" + str(TIME_STEPS) +  "_VARIANCE_SCHEDULE_[" + str(VARIANCE_SCHEDULE[0])  + ", " + str(VARIANCE_SCHEDULE[-1]) + "]" + "_EPOCHS_" + str(EPOCHS),
        config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "time_steps": TIME_STEPS,
        "variance_schedule": str(VARIANCE_SCHEDULE),
        "sample_rate": SAMPLE_RATE,
        "max_samples": MAX_SAMPLES,
        "with_conditional": WITH_CONDITIONING,
        "pred_noise": PRED_NOISE,
        "with_dropout": WITH_DROPOUT,
        }
    )

    model_filename = f'UNET_DIF_STEPS_{TIME_STEPS}_B_SIZE_{BATCH_SIZE}_LR_{LEARNING_RATE}_EPOCHS_{EPOCHS}_CONDITIONING_{WITH_CONDITIONING}'
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_validation',
        mode='min',
        dirpath=model_output_path,
        filename=model_filename,
        save_top_k=1,
        save_last=True,
    )

    #initialize datasets
    train_data = LatentsData(latents_dir=data_path, conditional_dir=conditional_path)
    val_data = LatentsData(latents_dir=val_data_path, conditional_dir=val_conditional_path)

    #initialize train dataloader
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        # shuffle=True,
        num_workers=24, 
        )
     #initialize train dataloader
    validationloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        # shuffle=True,
        num_workers=24, 
        )

    #initialize model
    model = DenoisingModel()
    lit_model = LitModel(model)

    #train model
    trainer = pl.Trainer(callbacks=[checkpoint_callback], max_epochs=EPOCHS, accelerator="auto", devices="auto", precision=16, logger=wandb_logger, check_val_every_n_epoch=1, val_check_interval=0.25)
    # trainer.fit(model=lit_model, train_dataloaders=trainloader, val_dataloaders=validationloader, ckpt_path=model_checkpoint)
    trainer.fit(model=lit_model, train_dataloaders=trainloader, ckpt_path=model_checkpoint)
    wandb.save(model_output_path + "/*")

if __name__ == '__main__':
    train()
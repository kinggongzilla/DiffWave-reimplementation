import torch

# params used in DiffWave paper for unconditional training in comments below
EPOCHS = 100
BATCH_SIZE = 8 #16
LEARNING_RATE = 1e-3 #2 * 1e-4
NUM_BLOCKS = 5 #36
RES_CHANNELS = 128 #256
TIME_STEPS = 50 #200
VARIANCE_SCHEDULE = torch.linspace(10e-4, 0.02, TIME_STEPS) #???
LAYER_WIDTH = 64 #512
SAMPLE_RATE = 22050 #44100 #16000
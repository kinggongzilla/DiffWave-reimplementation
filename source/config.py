import torch

# params used in DiffWave paper for unconditional training in comments below
EPOCHS = 5000
BATCH_SIZE = 1 #16
LEARNING_RATE = 2 * 1e-4
NUM_BLOCKS = 8 #36
RES_CHANNELS = 48 #256
LAYER_WIDTH = 64 #512
TIME_STEPS = 50 #200
VARIANCE_SCHEDULE = torch.linspace(10e-4, 0.02, TIME_STEPS) #???
SAMPLE_RATE = 16000 #44100 #22050
SAMPLE_LENGTH_SECONDS = 5
MAX_SAMPLES = 10000 # None for all samples in data input folder


# local config for laptop for debugging locally
# EPOCHS = 5
# BATCH_SIZE = 4 #16
# LEARNING_RATE = 1e-3 #2 * 1e-4
# NUM_BLOCKS = 2 #36
# RES_CHANNELS = 32 #256
# LAYER_WIDTH = 32 #512
# TIME_STEPS = 50 #200
# VARIANCE_SCHEDULE = torch.linspace(10e-4, 0.02, TIME_STEPS) #???
# SAMPLE_RATE = 16000 #44100 #22050
# SAMPLE_LENGTH_SECONDS = 5
# MAX_SAMPLES = None # None for all samples in data input folder
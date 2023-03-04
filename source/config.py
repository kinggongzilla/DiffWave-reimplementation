import torch

EPOCHS = 1
BATCH_SIZE = 6 #16
LEARNING_RATE = 2 * 1e-4
NUM_BLOCKS = 1 #36
RES_CHANNELS = 64 #256
TIMESTEP_LAYER_WIDTH = 512 #512
TIME_STEPS = 200
VARIANCE_SCHEDULE = torch.linspace(10e-4, 0.02, TIME_STEPS) #torch.linspace(10e-4, 0.05, TIME_STEPS)
SAMPLE_RATE = 8000 #16000 #22050 #44100
SAMPLE_LENGTH_SECONDS = 4
MAX_SAMPLES = 9000 #9000 # Use "None" for all samples in data input folder; 1000 = ~1h 6min
WITH_CONDITIONING=False

#CONFIG DATA PREP TRANSFORM TO MEL SPECTROGRAM
WINDOW_LENGTH=1024
HOP_LENGTH=256
N_FFT=1024
N_MELS=80
FMIN=20.0
FMAX=SAMPLE_RATE/2
POWER=1.0
NORMALIZED=True
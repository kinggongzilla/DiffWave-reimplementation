import torch

EPOCHS = 1000
BATCH_SIZE = 128
LEARNING_RATE = 2 * 1e-5
TIME_STEPS = 2
VARIANCE_SCHEDULE = torch.linspace(1e-5, 1, TIME_STEPS)
MAX_SAMPLES = 9000 #9000 # Use "None" for all samples in data input folder
WITH_CONDITIONING=True

#CONFIG DATA PREP 
SAMPLE_RATE = 44100 #16000 #22050 #44100

# TRANSFORM TO MEL SPECTROGRAM
WINDOW_LENGTH=1024
HOP_LENGTH=256
N_FFT=1024
N_MELS=128
FMIN=20.0
FMAX=SAMPLE_RATE/2
POWER=1.0
NORMALIZED=True
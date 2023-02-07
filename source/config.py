import torch

# params used in DiffWave paper for unconditional training in comments below
EPOCHS = 500
BATCH_SIZE = 8 #16
LEARNING_RATE = 2 * 1e-4
NUM_BLOCKS = 20 #36
RES_CHANNELS = 64 #256
TIMESTEP_LAYER_WIDTH = 512 #512
TIME_STEPS = 200 #200
VARIANCE_SCHEDULE = torch.linspace(10e-4, 0.05, TIME_STEPS) #torch.linspace(10e-4, 0.05, TIME_STEPS)
SAMPLE_RATE = 8000 #16000 #22050 #44100
SAMPLE_LENGTH_SECONDS = 4
MAX_SAMPLES = 4500 # Use "None" for all samples in data input folder


# local config for laptop for debugging locally
# EPOCHS = 5
# BATCH_SIZE = 4 #16
# LEARNING_RATE = 1e-3 #2 * 1e-4
# NUM_BLOCKS = 2 #36
# RES_CHANNELS = 32 #256
# TIMESTEP_LAYER_WIDTH = 32 #512
# TIME_STEPS = 50 #200
# VARIANCE_SCHEDULE = torch.linspace(10e-4, 0.02, TIME_STEPS) #???
# SAMPLE_RATE = 16000 #44100 #22050
# SAMPLE_LENGTH_SECONDS = 5
# MAX_SAMPLES = None # None for all samples in data input folder


#EXCERPT FROM PAPER REGARDING CONFIGURATION OF MODELS
# We compare DiffWave with several state-of-the-art neural vocoders, including WaveNet,
# ClariNet, WaveGlow and WaveFlow. Details of baseline models can be found in the original papers.
# Their hyperparameters can be found in Table 1. Our DiffWave models have 30 residual layers, kernel
# size 3, and dilation cycle [1, 2, · · · , 512]. We compare DiffWave models with different number of
# diffusion steps T ∈ {20, 40, 50, 200} and residual channels C ∈ {64, 128}. We use linear spaced
# schedule for βt ∈ [1 × 10−4
# , 0.02] for DiffWave with T = 200, and βt ∈ [1 × 10−4
# , 0.05] for
# DiffWave with T ≤ 50. The reason to increase βt for smaller T is to make q(xT |x0) close to platent.
# In addition, we compare the fast sampling algorithm with smaller Tinfer (see Appendix B), denoted as
# DiffWave (Fast), with the regular sampling (Algorithm 2). Both of them use the same trained models.
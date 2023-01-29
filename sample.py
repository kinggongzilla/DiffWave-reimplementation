import torch
from source.model import DiffWave

EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 2 * 1e-4
NUM_BLOCKS = 5
RES_CHANNELS = 16
TIME_STEPS = 50
VARIANCE_SCHEDULE = torch.linspace(0.001, 0.05, TIME_STEPS)
LAYER_WIDTH = 128

model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH)
model.load_state_dict(torch.load('model.pt'))
model.eval()

noise = torch.randn(1, 1, 22050*5)
model.sample(noise)

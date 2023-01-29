import torch
from source.model import DiffWave
from source.model_constants import NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH



model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH)
model.load_state_dict(torch.load('model.pt'))
model.eval()

noise = torch.randn(1, 1, 22050*5)
model.sample(noise)

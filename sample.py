import os
import torch
import torchaudio
from source.model import DiffWave
from source.model_constants import NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH



model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH)
model.load_state_dict(torch.load('outputs/models/model.pt'))
model.eval()

noise = torch.randn(1, 1, 22050*5) # 22KHz * 5000 milliseconds = 5 seconds of noise
y = model.sample(noise)
for i in range(y.shape[0]): #for each sample in batch
    path = os.path.join("outputs/samples", f"sample{i}.wav")
    torchaudio.save(path, y[i], 44100//2)
import os
import sys
import torch
import torchaudio
from source.model import DiffWave
from source.model_constants import NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS

model_path = "outputs/models/model.pt"

#get first commandline argument
if len(sys.argv) > 1:
    model_path = sys.argv[1]

model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH)
model.load_state_dict(torch.load(model_path))
model.eval()

noise = torch.randn(1, 1, SAMPLE_RATE*SAMPLE_LENGTH_SECONDS) # 22KHz * 5000 milliseconds = 5 seconds of noise
y = model.sample(noise)
for i in range(y.shape[0]): #for each sample in batch
    path = os.path.join("outputs/samples", f"sample{i}.wav")
    torchaudio.save(path, y[i], SAMPLE_RATE)
import os
import sys
import torch
import torchaudio
from source.model import DiffWave
from source.config import NUM_BLOCKS, RES_CHANNELS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = "output/models/model.pt"
if len(sys.argv) > 1:
    model_path = sys.argv[1]


model = DiffWave(RES_CHANNELS, NUM_BLOCKS, TIME_STEPS, VARIANCE_SCHEDULE, LAYER_WIDTH)
model.load_state_dict(torch.load(model_path))
model.eval()

noise = torch.randn(1, 1, SAMPLE_RATE*SAMPLE_LENGTH_SECONDS) # batch_size, n_channels, sample length e.g. 22,05KHz * 5000 milliseconds = 5 seconds of noise
y = model.sample(noise)
for i in range(y.shape[0]): #for each sample in batch
    path = os.path.join("output/samples", f"sample{i}.wav")
    torchaudio.save(path, y[i], SAMPLE_RATE)
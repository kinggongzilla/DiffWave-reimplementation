import os
import sys
from pydub import AudioSegment
from config import MAX_SAMPLES, SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, WINDOW_LENGTH, HOP_LENGTH, N_FFT, N_MELS, FMIN, FMAX, POWER, NORMALIZED, N_FFT, N_MELS, FMIN, FMAX, POWER, NORMALIZED, FMIN, FMAX, POWER, NORMALIZED
import torchaudio
import numpy as np
import torch
#Note: ffmpeg package required for pydub

def transform_to_spectrogram(
    audio_path: str, 
    out_path='data/mel_spectrograms', 
    sample_rate = SAMPLE_RATE, 
    win_length= WINDOW_LENGTH,
    hop_length= HOP_LENGTH,
    n_fft=N_FFT,
    f_min=FMIN,
    f_max = FMAX,
    n_mels = N_MELS,
    power = POWER,
    normalized = NORMALIZED,
):
    if not os.path.exists(out_path):
        raise 'out_dir does not exist'
    if not os.path.isfile(audio_path):
        raise 'given wav_path is not a file'

    filename = os.path.basename(audio_path)

    audio = torchaudio.load(audio_path)[0]
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=power,
        normalized=normalized)(audio)
    mel_spectrogram = 20 * torch.log10(torch.clamp(mel_spectrogram, min=1e-5)) - 20
    mel_spectrogram = torch.clamp((mel_spectrogram + 100) / 100, 0.0, 1.0)
    np.save(os.path.join(out_path, f'{filename}.spec.npy'), mel_spectrogram.cpu().numpy())



#load data of one wav and split it into chunks
def chop_wav(song_id: str, audio_path: str, out_dir: str, length: int):

    print('audio_path: ', audio_path)
    #check if out_dir exists
    if not os.path.exists(out_dir):
        raise 'out_dir does not exist'
    if not os.path.isfile(audio_path):
        raise 'given wav_path is not a file'

    #load mp3 audio from audio_path with torchaudio
    audio = torchaudio.load(audio_path)[0]
    audio = torchaudio.transforms.Resample(44100, 22050)(audio)

    #check if file ending is .mp3 or .wav
    file_ending = '.wav'
    if audio_path[-4:] == '.mp3':
        file_ending = '.mp3'

    start = 0
    end = length
    n_iters = int(len(audio[0])) // length

    for i in range(n_iters):
        #break if max samples reached
        if len(os.listdir(chopped_audio_out_path)) >= (MAX_SAMPLES):
            break
        newAudio = audio[:, start:end]
        #save newAudio with file_ending to out_dir with torchaudio
        torchaudio.save(os.path.join(out_dir, '{}_{}{}'.format(song_id, start, file_ending)), newAudio, SAMPLE_RATE)

        start += length
        end += length

if __name__ == '__main__':
    in_path=os.path.join('raw_samples')
    chopped_audio_out_path=os.path.join('data/chunked_audio')
    sample_length = SAMPLE_LENGTH_SECONDS * SAMPLE_RATE

    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    if len(sys.argv) > 2:
        chopped_audio_out_path = sys.argv[2]
    if len(sys.argv) > 3:
        sample_length = int(sys.argv[3])

    #loop over files in audio_folder_path
    for i, file in enumerate(os.listdir(in_path)):
        #break if max samples reached
        if len(os.listdir(chopped_audio_out_path)) >= (MAX_SAMPLES):
            break
        chop_wav(i, os.path.join(in_path, file), chopped_audio_out_path, sample_length)

    #generate mel spectrograms from chopped audio
    for i, file in enumerate(os.listdir(chopped_audio_out_path)):
        transform_to_spectrogram(os.path.join(chopped_audio_out_path, file), out_path='data/mel_spectrograms')
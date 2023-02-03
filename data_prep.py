import os
import sys
from pydub import AudioSegment
from source.model_constants import SAMPLE_RATE, SAMPLE_LENGTH_SECONDS
#Note: ffmpeg package required for pydub

#load data of one wav and split it into chunks
def chop_wav(song_id: str, audio_path: str, out_dir: str, length: int):
    print('audio_path: ', audio_path)
    #check if out_dir exists
    if not os.path.exists(out_dir):
        raise 'out_dir does not exist'
    if not os.path.isfile(audio_path):
        raise 'given wav_path is not a file'

    #load audio
    if audio_path.endswith('.wav'):
        audio = AudioSegment.from_wav(audio_path)
        file_ending = '.wav'
    elif audio_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_path)
        file_ending = '.mp3'
    else:
        raise 'wav_path must be a .wav or .mp3 file'

    start = 0
    end = length
    n_iters = int(len(audio)) // (SAMPLE_LENGTH_SECONDS * 1000)


    for i in range(n_iters):
        newAudio = audio[start:end]
        newAudio.export(os.path.join(out_dir, '{}_{}{}'.format(song_id, start, file_ending)), format="wav")
        start += length
        end += length

if __name__ == '__main__':
    in_path=os.path.join('raw_samples')
    out_path=os.path.join('chunked_audio')
    sample_length = SAMPLE_LENGTH_SECONDS * 1000 #milliseconds

    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    if len(sys.argv) > 2:
        out_path = sys.argv[2]
    if len(sys.argv) > 3:
        sample_length = int(sys.argv[3])

    #loop over files in audio_folder_path
    for i, file in enumerate(os.listdir(in_path)):
        chop_wav(i, os.path.join(in_path, file), out_path, sample_length)
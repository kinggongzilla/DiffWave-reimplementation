import os
from pydub import AudioSegment

#load data of one wav and split it into chunks
def chop_wav(song_id: str, wav_path: str, out_dir: str, length: int):

    #check if out_dir exists
    if not os.path.exists(out_dir):
        raise 'out_dir does not exist'
    if not os.path.isfile(wav_path):
        raise 'given wav_path is not a file'

    audio = AudioSegment.from_wav(wav_path)

    n_iters = int(len(audio) / 5000)

    start = 0
    end = length

    for i in range(n_iters):
        newAudio = audio[start:end]
        newAudio.export('{}{}_{}.wav'.format(out_dir, song_id, start), format="wav")
        start += length
        end += length


#TODO: Download data function
def downlad_dataset(url: str, path_tsv: str):
    username = 'marcopasini'
    password = 'letmetrainonthis'
    # with open(path_tsv) as f: 
    # f.readline()  # skip header 
    # for l in f: 
    #     track, artist, album, _, _, _, tags = l.rstrip('\r\n').split('\t') 
    #     if 'techno:' in tags: 
    #         url = 'https://teacap.cp.jku.at/protected/edm-datasets/cp-jamendo/audio/%s/%s/%s.mp3' % (artist, album, track) 

            # requests.get('url/{artist}/{album}/{}track'.format(subpath=filenamelist[i]), auth=(username, password))

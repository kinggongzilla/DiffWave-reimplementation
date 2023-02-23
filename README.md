# DiffWave-implementation
A custom DiffWave implementation for the Practical Work in AI course at JKU. 

# Set up environment with dependencies

2. Load conda env with "conda create -n <environment-name> --file req.txt"
3. Activate conda environment with "conda activate <environment-name>"
4. If on Mac/Linux install sox with "conda install -c conda-forge sox" (a dependency of torchaudio)

# Prepare data
By default, full length audio files are chunked into fixed length samples of 4 seconds (length is configurable). Mel spectrograms are computed for each sample.
Default input folder for data processing is "raw_samples"
Default output folder for chunked audio is "data/chunked_audio"
Default output folder for mel spectrograms is "data/mel_spectrograms"

1. Make sure that "data/chunked_audio" and "data/mel_spectrograms" folders exist.
2. Run "python data_prep.py". Note that the input folder should only contain .mp3 or .wav files of equal length.
3. Optional: To pass different input/output folders run "python data_prep.py [path to audio_folder] [path to output_folder]"

# How to train a model
All samples used for training have to be of the SAME length and in the same folder. Samples have to be either .mp3 or .wave .mp3. 
1. Set config parameters in "source/config.py"
2. Run "python main.py [path to data_folder] [path to conditional input (i.e. spectrograms)]" to start training. Note that data_folder should only contain .mp3 or .wav files of equal length. The default paths are "data/chunked_audio" and "data/mel_spectrograms"


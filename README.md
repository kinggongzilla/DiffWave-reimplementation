# DiffWave-implementation
A custom DiffWave implementation I am doing in the course of the Practical Work in AI course at JKU. 

# How to execute

1. Put all .mp3 or .wav files of SAME length into a folder. data_prep.py can be used to chop up .mp3 files into .wav files of equal length. "python data_prep.py [path to full length audio folder] [path to output folder] [length of output files in seconds]"
2. Load conda env with "conda env create --name envname --file=environments.yml"
3. Activate conda environment with "conda activate envname"
4. If on Mac/Linux install sox with "conda install -c conda-forge sox"
5. Run "python main.py [path to data_folder]"


data_folder should be one folder containing .mp3 files of equal length.

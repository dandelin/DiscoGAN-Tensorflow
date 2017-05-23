import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from tfrecordreadwrite import convert_to, read_and_decode
from os import walk, mkdir
from glob import glob


SAMPLING_RATE = 16000
FFT_SIZE = 1024 #Frequency resolution
hop_length = int(FFT_SIZE/6.5)
eps = np.finfo(float).eps

#FIX ME : write your data directory
male_dir_path = "./Male"

temp = []
audio_male_path = []

for (dirpath, dirnames, filenames) in walk(male_dir_path):
    temp.append(dirpath)

del(temp[0])

for path in temp: 
    audio_path = glob(path+"/*.wav")
    audio_male_path.extend(audio_path)



#FIX ME : write your data directory
female_dir_path = "./Female"

temp = []
audio_female_path = []

for (dirpath, dirnames, filenames) in walk(female_dir_path):
    temp.append(dirpath)

del(temp[0])

for path in temp: 
    audio_path = glob(path+"/*.wav")
    audio_female_path.extend(audio_path)



spectrogram_male_save = []
spectrogram_female_save = []

i = 0
threshold = -4000
frame = 0
duration = 2.5
#speech_male_file fetch from directory
for audio in audio_male_path:
    i += 1
    
    
    y, sr = librosa.core.load(audio, sr = SAMPLING_RATE, mono=True)
    if len(y) < duration*SAMPLING_RATE:
       continue

    k = int(len(y)/1024)
    print("ylength_before",len(y))
    for j in range(k):
        sum = np.sum(np.log10(np.power(y[frame:frame+1024],2)))
        print(j)
        print("sum",sum)
        if sum > threshold :
            onset = frame
            print("onset", onset)
            break
        frame += 1024

    frame = 0

    try:
        y = y[onset:onset+int(duration*SAMPLING_RATE)]
        print("yvalue", y[int(duration*SAMPLING_RATE)-1])
    except IndexError:
        print("except")
        continue

    print("ylength_after",len(y))
    print("i",i)
    D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
    D = np.log10(np.abs(D)+eps) # Magnitude of plain spectrogram
    # D = librosa.feature.melspectrogram(y=y, n_fft=FFT_SIZE, hop_length=hop_length, sr=sr, n_mels=128, fmax=None) # use when you want to use mel-spectrogram

    D = np.expand_dims(D, axis=2)
    spectrogram_male_save.append(D)

print("Save male spec array")
spectrogram_male_save = np.asarray(spectrogram_male_save, dtype=np.float32)
print("Male spec shape", spectrogram_male_save.shape)


#convert numpy array(male_spectrogram) to tf_recordfile
print("Male spec converting start")

try:
    mkdir("./log_spectrograms_male")
    convert_to(spectrogram_male_save, "./log_spectrograms_male/log_spectrograms_male_speech.tfrecords")
except OSError :
    convert_to(spectrogram_male_save, "./log_spectrograms_male/log_spectrograms_male_speech.tfrecords")
    
print("Male spec converting finished")


i = 0
frame = 0
#speech_female_file fetch from directory
for audio in audio_female_path:
    i += 1
    
    
    y, sr = librosa.core.load(audio, sr = SAMPLING_RATE, mono=True)
    if len(y) < duration*SAMPLING_RATE:
        continue
    k = int(len(y)/1024)
    
    for j in range(k):
        sum = np.sum(np.log10(np.power(y[frame:frame+1024],2)))
        if sum > threshold :
            onset = frame
            print("onset",onset)
            break
        frame += 1024

    frame = 0

    try:
        y = y[onset:onset+int(duration*SAMPLING_RATE)]
        print("yvalue", y[int(duration*SAMPLING_RATE)-1])
    except IndexError:
        print("except")
        continue

    print(len(y))
    print(i)

    D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
    D = np.log10(np.abs(D)+eps) # Magnitude of plain spectrogram
   
   
    # D = librosa.feature.melspectrogram(y=y, n_fft=FFT_SIZE, hop_length=hop_length, sr=sr, n_mels=128, fmax=None) # use when you want to use mel-spectrogram
    D = np.expand_dims(D, axis=2)
    spectrogram_female_save.append(D)

print("Save female spec array")
spectrogram_female_save = np.asarray(spectrogram_female_save, dtype=np.float32)
print("Female spec shape", spectrogram_female_save.shape) #(Numberofspecs,col,row)


#convert numpy array(female_spectrogram) to tf_recordfile
print("Female spec converting start")

try:
    mkdir("./log_spectrograms_female")
    convert_to(spectrogram_female_save, "./log_spectrograms_female/log_spectrograms_female_speech.tfrecords")
except OSError :
    convert_to(spectrogram_female_save, "./log_spectrograms_female/log_spectrograms_female_speech.tfrecords")
     
print("Female spec converting finished")

# D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
# display.specshow(librosa.power_to_db(D, ref=np.max), y_axis='log', fmax=None, x_axis='time')




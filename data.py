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
hop_length = int(FFT_SIZE/8)


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

#speech_male_file fetch from directory
for audio in audio_male_path:
    i += 1
    print(i)
    #loading from offset(1s) to (4s)
    y, sr = librosa.core.load(audio, sr = SAMPLING_RATE, mono=True, offset=1, duration=3)
    
    if len(y)<=3*SAMPLING_RATE:
        pass

    D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
    D = np.abs(D) # Magnitude of plain spectrogram
    # D = librosa.feature.melspectrogram(y=y, n_fft=FFT_SIZE, hop_length=hop_length, sr=sr, n_mels=128, fmax=None) # use when you want to use mel-spectrogram
    D = np.expand_dims(D, axis=2)
    spectrogram_male_save.append(D)

spectrogram_male_save = np.asarray(spectrogram_male_save, dtype=np.float32)
print("Male_spec_shape", spectrogram_male_save.shape)


#convert numpy array(male_spectrogram) to tf_recordfile
try:
    mkdir("./spectrograms_male")
    convert_to(spectrogram_male_save, "./spectrograms_male/spectrograms_male_speech.tfrecords")
except OSError :
    convert_to(spectrogram_male_save, "./spectrograms_male/spectrograms_male_speech.tfrecords")
    



i = 0

#speech_female_file fetch from directory
for audio in audio_female_path:
    i += 1
    print(i)
    #loading from offset(1s) to (4s)
    y, sr = librosa.core.load(audio, sr = SAMPLING_RATE, mono=True, offset=1, duration=3)
    D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
    D = np.abs(D) # Magnitude of plain spectrogram
    # D = librosa.feature.melspectrogram(y=y, n_fft=FFT_SIZE, hop_length=hop_length, sr=sr, n_mels=128, fmax=None) # use when you want to use mel-spectrogram
    D = np.expand_dims(D, axis=2)
    spectrogram_female_save.append(D)

spectrogram_female_save = np.asarray(spectrogram_female_save, dtype=np.float32)
print("Female_spec_shape", spectrogram_female_save.shape) #(Numberofspecs,col,row)


#convert numpy array(female_spectrogram) to tf_recordfile
try:
    mkdir("./spectrograms_female")
    convert_to(spectrogram_female_save, "./spectrograms_female/spectrograms_female_speech.tfrecords")
except OSError :
    convert_to(spectrogram_female_save, "./spectrograms_female/spectrograms_female_speech.tfrecords")
     


D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
display.specshow(librosa.power_to_db(D, ref=np.max), y_axis='log', fmax=None, x_axis='time')


#D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
# plt.title('Spectrogram')
# plt.imshow(np.log10(D+0.1), aspect = 'auto')
# plt.show()
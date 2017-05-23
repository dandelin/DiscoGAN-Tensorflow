import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import os

from tfrecordreadwrite import convert_to, read_and_decode
from os import walk, mkdir
from glob import glob


SAMPLING_RATE = 16000
FFT_SIZE = 1024 #Frequency resolution
hop_length = int(FFT_SIZE/6.5)
offset = 0
duration = 2.5



def search(dirname):
    path_list = glob("%s/*.wav"%(dirname))
    return path_list


#Get wav file path
bass_path_list = search("./bass")
keyboard_path_list = search("./keyboard")






spectrogram_bass_save = []
spectrogram_keyboard_save = []


#bass_file fetch from directory
for i, audio in enumerate(bass_path_list):
    print("{}th audio : ".format(i), audio)
    
    y, sr = librosa.core.load(audio, sr = SAMPLING_RATE, mono=True, offset = offset, duration = duration)

    D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
    D = np.abs(D) # Magnitude of plain spectrogram
    D = np.expand_dims(D, axis=2)
    spectrogram_bass_save.append(D)

print("Save bass_electronic spectrogram array")
spectrogram_bass_save = np.asarray(spectrogram_bass_save, dtype=np.float32)
print("Bass spectrogram shape", spectrogram_bass_save.shape)


#convert numpy array(spectrogram_bass_save) to tf_recordfile
print("Bass_electronic spectrogram converting start")

os.makedirs("./spectrograms_bass", exist_ok=True)
convert_to(spectrogram_bass_save, "./spectrograms_bass/spectrograms_bass.tfrecords")
    
print("Bass_electronic spectrogram converting finished")




#keyboard_file fetch from directory
for i, audio in enumerate(keyboard_path_list):
    
    print("{}th audio : ".format(i), audio)
    
    y, sr = librosa.core.load(audio, sr = SAMPLING_RATE, mono=True, offset = offset, duration = duration)

    D = librosa.stft(y=y, n_fft=FFT_SIZE, hop_length=hop_length, center=True) # win_length = FFT_SIZE
    D = np.abs(D) # Magnitude of plain spectrogram
    D = np.expand_dims(D, axis=2)
    spectrogram_keyboard_save.append(D)

print("Save keyboard_electronic spectrogram array")
spectrogram_keyboard_save = np.asarray(spectrogram_keyboard_save, dtype=np.float32)
print("Keyboard spec shape", spectrogram_keyboard_save.shape)


#convert numpy array(spectrogram_keyboard_save) to tf_recordfile
print("Keyboard_electronic spectrogram converting start")

os.makedirs("./spectrograms_keyboard", exist_ok=True)
convert_to(spectrogram_keyboard_save, "./spectrograms_keyboard/spectrograms_keyboard.tfrecords")
    
print("Keyboard_electronic spectrogram converting finished")



import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
from os import walk

domain_A_path = "/Users/Adrian/Desktop/Domain_A"
domain_B_path = "/Users/Adrian/Desktop/Domain_B"
domain_A_file_list = []
domain_B_file_list = []

for (dirpath, dirnames, filenames) in walk(domain_A_path):
    domain_A_file_list.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(domain_B_path):
    domain_B_file_list.extend(filenames)
    break
    
print "domain_A_file_list : ", domain_A_file_list, "\n", "domain_B_file_list : ", domain_B_file_list



Spectrogram_A_save = []
Spectrogram_B_save = []

#domain_A_file fetch from directory
for audio in domain_A_file_list :

    #loading from offset(5s) to duration(8s)
    y, sr = librosa.core.load("/Users/Adrian/Desktop/Domain_A/" + audio, sr = 22050, \
     mono=True, offset=5.0, duration=3)

    scope_y = y/max(y)  # crop and normalize

    D = librosa.core.stft(scope_y, n_fft=2048, hop_length=1024, win_length=None, window='hann', \
     center=True)
    D = np.abs(D) #plain spectrogram
    S = librosa.feature.melspectrogram(y=scope_y, n_fft=2048, hop_length=1024, sr=sr, n_mels=128, fmax=None)#mel-spectrogram

    Spectrogram_A_save.append(S)
    

#domain_B_file fetch from directory
for audio in domain_A_file_list :

    #loading from offset(5s) to duration(8s)
    y, sr = librosa.core.load("/Users/Adrian/Desktop/Domain_A/" + audio, sr = 22050, \
     mono=True, offset=5.0, duration=3)

    scope_y = y/max(y)  # crop and normalize

    D = librosa.core.stft(scope_y, n_fft=2048, hop_length=1024, win_length=None, window='hann', \
     center=True)
    D = np.abs(D) #plain spectrogram
    S = librosa.feature.melspectrogram(y=scope_y, n_fft=2048, hop_length=1024, sr=sr, n_mels=128, fmax=None)#mel-spectrogram

    Spectrogram_B_save.append(S)

print "Spectrogram Shape : ", np.shape(Spectrogram_A_save[0])
print "Number of Spectrograms in Domain_A : ", len(Spectrogram_A_save)
print "Number of Spectrograms in Domain_B : ", len(Spectrogram_B_save)



plt.figure(figsize=(10, 4))
#librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=None, x_axis='time')
plt.title('Mel spectrogram')
plt.imshow(np.log10(Spectrogram_A_save[0]+0.1), aspect = 'auto')
plt.show()
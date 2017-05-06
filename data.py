import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
from tfrecordreadwrite import convert_to, read_and_decode
from os import walk, mkdir


#FIX ME : write your data directory
domain_A_path = "/Users/Adrian/Desktop/Domain_A"
domain_B_path = "/Users/Adrian/Desktop/Domain_B"
domain_A_file_list = []
domain_B_file_list = []


SAMPLING_RATE = 16000
FFT_SIZE = 1024 #Frequency resolution

for (dirpath, dirnames, filenames) in walk(domain_A_path):
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    domain_A_file_list.extend(filenames)
    break

for (dirpath, dirnames, filenames) in walk(domain_B_path):
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    domain_B_file_list.extend(filenames)
    break
    
print("domain_A_file_list : ", domain_A_file_list, "\n", "domain_B_file_list : ", domain_B_file_list)



Spectrogram_A_save = []
Spectrogram_B_save = []

#domain_A_file fetch from directory
for audio in domain_A_file_list :

    #loading from offset(5s) to duration(8s)
    y, sr = librosa.core.load("/Users/Adrian/Desktop/Domain_A/" + audio, sr = SAMPLING_RATE, mono=True, offset=5.0, duration=3)
    D = librosa.core.stft(y=y, n_fft=FFT_SIZE, hop_length=int(FFT_SIZE/2), win_length=None, window='hann', center=True) # win_length = FFT_SIZE
    D = np.abs(D) # Magnitude of plain spectrogram
    # D = librosa.feature.melspectrogram(y=scope_y, n_fft=2048, hop_length=1024, sr=sr, n_mels=128, fmax=None) # use when you want to use mel-spectrogram
    D = np.expand_dims(D, axis=2)
    # D = np.reshape(D,(1,-1,1)) #The last index indicates channel
    
    Spectrogram_A_save.append(D)
    # Spectrogram_A_save.extend(D)
    

#domain_B_file fetch from directory
for audio in domain_B_file_list :
    #loading from offset(5s) to duration(8s)
    y, sr = librosa.core.load("/Users/Adrian/Desktop/Domain_B/" + audio, sr = SAMPLING_RATE, mono=True, offset=5.0, duration=3)
    D = librosa.core.stft(y =y, n_fft=FFT_SIZE, hop_length=int(FFT_SIZE/2), win_length=None, window='hann', center=True)
    D = np.abs(D) # Magnitude of plain spectrogram
    # D = librosa.feature.melspectrogram(y=scope_y, n_fft=2048, hop_length=1024, sr=sr, n_mels=128, fmax=None)#mel-spectrogram
    D = np.expand_dims(D, axis=2)
    # D = np.reshape(D,(1,-1,1))
    
    
    Spectrogram_B_save.append(D)
    # Spectrogram_A_save.extend(D)
    


Spectrogram_A_save = np.array(Spectrogram_A_save, dtype = np.float32)
Spectrogram_B_save = np.array(Spectrogram_B_save, dtype = np.float32)



print("SHAPE_A : " , Spectrogram_A_save.shape) #(Numberofspecs,col,row)
print("SHAPE_B : " , Spectrogram_B_save.shape)




try:
    convert_to(Spectrogram_A_save, "./spectrogram_files_A/spectrograms_A.tfrecords")
    convert_to(Spectrogram_A_save, "./spectrogram_files_B/spectrograms_B.tfrecords")
except IOError :
    mkdir("./spectrogram_files_A")
    mkdir("./spectrogram_files_B")
    convert_to(Spectrogram_A_save, "./spectrogram_files_A/spectrograms_A.tfrecords")
    convert_to(Spectrogram_A_save, "./spectrogram_files_B/spectrograms_B.tfrecords")





# try:
#     for i, A in enumerate(Spectrogram_A_save):
#         convert_to(A, "./spectrogram_files_A/spectrograms_A_{}.tfrecords".format(i))
#     for i, B in enumerate(Spectrogram_B_save):      
#         convert_to(B, "./spectrogram_files_B/spectrograms_B_{}.tfrecords".format(i))
# except IOError :
#     mkdir("./spectrogram_files_A")
#     mkdir("./spectrogram_files_B")
#     for i, A in enumerate(Spectrogram_A_save):
#         convert_to(A, "./spectrogram_files_A/spectrograms_A_{}.tfrecords".format(i))
#     for i, B in enumerate(Spectrogram_B_save):
#         convert_to(B, "./spectrogram_files_B/spectrograms_B_{}.tfrecords".format(i))




# plt.figure(figsize=(10, 4))
#librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=None, x_axis='time')
# plt.title('Spectrogram')
# plt.imshow(np.log10(Spectrogram_A_save[0]+0.1), aspect = 'auto')
# plt.show()
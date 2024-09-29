import os

import librosa
import numpy as np
from matplotlib import pyplot as plt
import librosa.display
path1 = r"C:\Users\xrw\Desktop\sample_audio\c1"
path2 = r"C:\Users\xrw\Desktop\sample_audio\c2"

note_list1 = os.listdir(path1)
note_list2 = os.listdir(path2)



path = r"C:\Users\xrw\Desktop\cluster_audio\4_14_10_5cluster\cluster_0\nguru-1-E-3.wav"
y, sr = librosa.load(path)
n_fft = 4096
hop_length = 1024
stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(stft,
                                                       ref=np.max),
                               y_axis='log', x_axis='time', ax=ax)
fig.colorbar(img, ax=ax, format="%+2.0f dB")


# for i in note_list1:
#     print(i)
#     note_path = os.path.join(path1, i)
#     y, sr = librosa.load(note_path)
#     n_fft = 2048
#     hop_length = 512
#     fig, ax = plt.subplots()
#     S = librosa.magphase(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=np.ones, center=False))[0]
#     # print(S.shape)
#     times = librosa.times_like(S, n_fft=n_fft, hop_length=hop_length)
#     rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
#     rms = np.squeeze(rms)
#     ax.set_title("cluster1_" + i)
#     ax.plot(times, rms)
#
# for i in note_list2:
#     print(i)
#     note_path = os.path.join(path2, i)
#     y, sr = librosa.load(note_path)
#     n_fft = 2048
#     hop_length = 512
#     fig, ax = plt.subplots()
#     S = librosa.magphase(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=np.ones, center=False))[0]
#     # print(S.shape)
#     times = librosa.times_like(S, n_fft=n_fft, hop_length=hop_length)
#     rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
#     rms = np.squeeze(rms)
#     ax.set_title("cluster2_" + i)
#     ax.plot(times, rms)

plt.show()


import csv
import os

import librosa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from newdataset.maori import preprocessing


def ex_mfcc(dir):
    with open(r"C:\Users\邢\Desktop\10 nguru recording\mfcc.csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        dataset = np.array([])
        instrument_num = 1
        path_list = []
        for i in os.listdir(dir):
            file = os.path.splitext(i)
            if (file[1] == ".wav" or file[1] == ".mp3"):
                path_list.append(i)
        path_list.sort(key=lambda x:int(x[6:][:-4]))


        for i in path_list:
            file = os.path.splitext(i)
            # if (file[1] != ".wav" and file[1] != ".mp3"):
            #     continue;
            print(file)
            y, sr = librosa.load(os.path.join(dir, i), sr=None)
            length = len(y)

            n_fft = frame_length = 2048
            hop_length = int(n_fft / 4)
            fk = np.linspace(0, sr, n_fft)
            fk = fk[0:(len(fk) // 2) + 1]
            final_interval, win_interval, attack, five_part = preprocessing.preprecessing1(y, sr, 512, 40)

            S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, n_fft=n_fft,
                                                   hop_length=hop_length)
            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            S, phase = librosa.magphase(stft)
            S_abs = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
            S_mel = librosa.feature.melspectrogram(S = stft, sr = sr, n_fft = n_fft, hop_length = hop_length)

            for ele in win_interval:
                S_mel_p = librosa.power_to_db(S_mel[:, ele[0]:ele[1]])
                mfcc = librosa.feature.mfcc(S=S_mel_p, sr=sr)
                avg_mfcc = np.average(mfcc, axis=1)
                sample = np.array(avg_mfcc)
                print("mfcc: ", sample.shape)
                print(sample)
                w.writerow(sample)

def show_mfcc(dir):
    mfcc = pd.read_csv(mfcc_dir)
    print(mfcc)
    sns.pairplot(mfcc)
    plt.savefig(r"C:\Users\邢\Desktop\mfcc")
    plt.show()



if __name__ == '__main__':
    dir = r"C:\Users\邢\Desktop\10 nguru recording"
    dir1 = r"C:\Users\邢\Desktop\2recording"
    mfcc_dir = r"C:\Users\邢\Desktop\10 nguru recording\mfcc.csv"
    # ex_mfcc(dir)
    show_mfcc(mfcc_dir)



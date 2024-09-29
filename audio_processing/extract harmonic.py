import os

import librosa
import numpy
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import pathlib as Path
import feature_extraction
import preprocessing

csv_path1 = r'C:\Users\邢\Desktop\interval_csv\n76800.csv'
wav_path1 = r'C:\Users\邢\Desktop\interval_wav\n76800.wav'
csv_path2 = r'C:\Users\邢\Desktop\interval_csv\n310784.csv'
wav_path2 = r'C:\Users\邢\Desktop\interval_wav\n310784.wav'

def read_data(path):
    print(path)
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    data = data.reshape(data.shape[0], )
    print("data",data.shape)
    data = np.array(data)
    multi_f0 = np.zeros((data.shape[0],5))
    for i, str in enumerate(data):
        list = str.split("\t")
        for j in range(len(list)):
            multi_f0[i][j] = list[j]
        # print(multi_f0[i])
    return multi_f0

# def get_spec_cencentoid(path):

def piptrack(path):
    n_fft = 4096
    hop_length = int(n_fft/4)
    y, sr = librosa.load(path, sr=None)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
    print(pitches.shape)
    print(pitches)
    print(magnitudes.shape)
    print(magnitudes)


def reduce_outliers(data_set, q):
    data = data_set[:,0];
    print("contain nan: ", data.shape[0])
    boolmat = ~np.isnan(data)
    data = data[boolmat]
    data_set = data_set[boolmat]
    iqr = np.quantile(data,0.75) - np.quantile(data,0.25);
    val_low = np.quantile(data,0.75) - iqr * q
    val_up = np.quantile(data,0.25) + iqr * q
    normal_val = data_set[(data < val_up) & (data > val_low)]
    deleted_num = data.shape[0] - normal_val.shape[0]
    print("val_low: ", val_low, ", val_up: ",val_up)
    print("original ", data.shape[0], ", deleted: ", deleted_num)
    return normal_val


def get_f0(path):
    multi_f0 = read_data(path).T
    print("multi_f0",multi_f0.shape)
    times = multi_f0[0]
    fir_f0 = multi_f0[1]
    sec_f0 = multi_f0[2]
    trd_f0 = multi_f0[3]
    four_f0 = multi_f0[4]
    for i in range(len(fir_f0)):
        if(fir_f0[i] == 0):
            fir_f0[i] = np.nan
    for i in range(len(sec_f0)):
        if(sec_f0[i] == 0):
            sec_f0[i] = np.nan
    for i in range(len(trd_f0)):
        if(trd_f0[i] == 0):
            trd_f0[i] = np.nan
    for i in range(len(four_f0)):
        if(four_f0[i] == 0):
            four_f0[i] = np.nan
    # for i in multi_f0:
    #     i[0] = int(round(i[0]/time_sr))
    # fir_f0, sec_f0, trd_f0 = np.zeros(length)
    # for i in multi_f0.shape[0]:
    #     seq = multi_f0[i][0]
    #     if(fir_f0[seq] == 0):
    #       fir_f0[seq] = multi_f0[i][1]
    #     else:
    #       fir_f0[seq] = multi_f0[i][1]+fir_f0[seq]/2
    #     if (sec_f0[seq] == 0):
    #         sec_f0[seq] = multi_f0[i][2]
    #     else:
    #         sec_f0[seq] = multi_f0[i][2] + sec_f0[seq] / 2
    #     if (trd_f0[seq] == 0):
    #         trd_f0[seq] = multi_f0[i][3]
    #     else:
    #         trd_f0[seq] = multi_f0[i][3] + trd_f0[seq] / 2
    return fir_f0,sec_f0,trd_f0, four_f0, times
        # if (fth_f0[seq] == 0):
        #     fth_f0[seq] = multi_f0[i][4]
        # else:
        #     fth_f0[seq] = multi_f0[i][4] + fth_f0[seq] / 2

def print_set(set):
    for j in range(0, set.shape[0], 100):
        print(j, " ", set[j])

def get_num(length,f0, fre_sr):
    num_freq = np.zeros(length,dtype= int)
    for i in range(length):
        if (np.isnan(f0[i])):
            pass
        else:
            num_freq[i] = int(round(f0[i] / fre_sr));
    return num_freq


def get_energy(f0, S, num, harmonic):
    threshold = 1;
    energy_array = np.zeros((S.shape[1], harmonic.shape[0]+1))
    harmonic_to_f0 = np.zeros((S.shape[1],harmonic.shape[0]-1))
    for i in range(S.shape[1]):
        energy_array[i][0] = f0[i];
        if (num[i] == 0):
            pass
        else:
            for j in range(1, harmonic.shape[0]+1,1):
                for m in range(-threshold + 1, threshold, 1):
                    # print(S[(num[i] + 1) * harmonic[j] - 1 + m][i])
                    maxenergy = S[(num[i] + 1) * harmonic[j - 1] - 1 + m][i];
                    if(energy_array[i][j] < maxenergy):
                        energy_array[i][j] = maxenergy;
                if (j >= 2):
                    harmonic_to_f0[i][j - 2] = energy_array[i][j] / energy_array[i][1]
    return energy_array, harmonic_to_f0

def extract_feature(dir):
    for i in os.listdir(dir):
        file = os.path.splitext(i)
        if (file[1] == ".wav"):
            y, sr = librosa.load(os.path.join(dir, i), sr=None)
            newfile = file[0] + ".csv"
            # y, sr = librosa.load(r"C:\Users\邢\Desktop\new_violin_audio", sr=None)
            n_fft = frame_length = 4096
            fre_sr = sr / n_fft
            print(y.shape)
            print("sr", sr)

            # f0, voiced_flag, voiced_probs = librosa.pyin(y, frame_length=n_fft, fmin=196,
            #                                              fmax=librosa.note_to_hz('C7'), sr=sr)
            # f0 = np.nan_to_num(f0)
            # freq_num = get_num(f0.shape[0], f0, fre_sr)
            fir_f0, sec_f0, trd_f0, times = get_f0(os.path.join(dir, newfile), sr, n_fft)
            stft = np.abs(librosa.stft(y, n_fft=n_fft))
            # harmonic = np.array([1, 2, 3, 4, 5, 6])
            # print(freq_num.dtype)
            # energy_set, harmonic_set = get_energy(stft, freq_num, harmonic)
            # for ele in energy_set:
            #     print(ele)
            # # for ele in harmonic_set:
            # #     print(ele)
            # print(energy_set.shape)
            # print(harmonic_set.shape)
            # # print(f0)
            # print(f0.shape)
            # times = librosa.times_like(f0, sr=sr, n_fft=n_fft, hop_length=n_fft / 4)
            print("time", times)
            D = librosa.amplitude_to_db(stft, ref=np.max)
            print(D.shape)
            print(fre_sr)
            fig, ax = plt.subplots()
            img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=sr, n_fft=n_fft,
                                           hop_length=n_fft / 4)
            ax.set(title=i + 'stft')
            fig.colorbar(img, ax=ax, format="%+2.f dB")
            ax.plot(times, fir_f0, label='fir_f0', color='cyan', linewidth=1)
            ax.plot(times, sec_f0, label='sec_f0', color='g', linewidth=1)
            ax.plot(times, trd_f0, label='trd_f2', color='w', linewidth=1)
            # ax.plot(times, 2*fir_f0, label='fir_f0_har1', color='r', linewidth=1)
            # ax.plot(times, 2*sec_f0, label='sec_f0_har1', color='b', linewidth=1)
            # ax.plot(times, 2*trd_f0, label='trd_f0_har1', color='w', linewidth=1)

            # ax.plot(times, f0, label='f0', color='cyan', linewidth=1)
            # ax.plot(times, 2*f0, label='f1', color='w', linewidth=1)
            # ax.plot(times, 3*f0, label='f2', color='b', linewidth=1)
            # ax.plot(times, 4*f0, label='f3', color='g', linewidth=1)
            # ax.plot(times, 5*f0, label='f4', color='cyan', linewidth=1)
            # ax.plot(times, 6*f0, label='f5', color='w', linewidth=1)
            # ax.plot(times, 7*f0, label='f6', color='cyan', linewidth=2)
            ax.legend(loc='upper right')
            plt.show()


def show_multi_f0(csvp, wavp):
    n_fft = 1024
    hop_length = 256
    fir_f0,sec_f0,trd_f0, four_f0, times = get_f0(csvp)
    print("fir_f0: ", fir_f0.shape)
    print(fir_f0)
    y,sr = librosa.load(wavp, sr = 22050)
    print(sr)
    S = librosa.stft(y = y, n_fft = n_fft, hop_length = hop_length)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, frame_length=n_fft, hop_length = hop_length, fmin=librosa.note_to_hz('G3'),
                                                 fmax=librosa.note_to_hz('C7'), sr=sr)
    print("f0: ", f0.shape)
    print(f0)
    S = np.abs(S)
    print(S)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(times, fir_f0, label='fir_f0', color='cyan', linewidth=1)
    ax[0].plot(times, sec_f0, label='sec_f0', color='r', linewidth=1)
    ax[0].plot(times, trd_f0, label='trd_f0', color='g', linewidth=1)
    ax[0].plot(times, four_f0, label='four_f0', color='w', linewidth=1)
    ax[0].set(xticks=[])
    ax[0].legend(loc='upper right')
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[0], n_fft = n_fft, hop_length = hop_length)
    ax[0].set(title= 'multi_f0')
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1], n_fft=n_fft, hop_length=hop_length)
    new_times = librosa.times_like(f0, sr=sr, n_fft=n_fft, hop_length=hop_length)
    print("multi_time: ", times.shape)
    print(times)
    print("new_times: ", new_times.shape)
    print(new_times)
    # ax[1].set(title='pyin f0')
    # ax[1].plot(new_times, f0, label='fir_f0', color='cyan', linewidth=1)

    new_f0 = fir_f0[2:]
    final_f0 = np.zeros(f0.shape)
    for i in range(new_f0.shape[0]):
        front = 5
        back = 5
        len = front+ back
        total = 0
        max = 0
        min = 5000
        if(np.isnan(new_f0[i])):
            if (np.isnan(f0[i])):
                final_f0[i] = f0[i]
                continue
            if (i < front or i >= new_f0.shape[0] - back):
                final_f0[i] = f0[i]
                continue
            else:
                for j in range(i - front, i + back, 1):
                    if (np.isnan(new_f0[j])):
                        len = len - 1
                    else:
                        if(new_f0[j] > max):
                            max = new_f0[j]
                        if(new_f0[j] < min):
                            min = new_f0[j]
                        total += new_f0[j]
                if (len == 0):
                    final_f0[i] = f0[i]
                    continue
                avg = total/len
            if((np.abs(f0[i] - avg)) < 100 or min < f0[i] < max):
                final_f0[i] = f0[i]
            else:
                final_f0[i] = new_f0[i]
        else:
            final_f0[i] = new_f0[i]
    ax[1].set(title='updated first f0')
    ax[1].plot(new_times, final_f0, label='fir_f0', color='cyan', linewidth=1)
    plt.show()

def show_ori(wavp):
    n_fft = 1024
    hop_length = 256
    y,sr = librosa.load(wavp, sr = 22050)
    print(sr)
    S = librosa.stft(y = y, n_fft = n_fft, hop_length = hop_length)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, frame_length=n_fft, hop_length = hop_length, fmin=librosa.note_to_hz('G3'),
                                                 fmax=librosa.note_to_hz('C7'), sr=sr)
    S = np.abs(S)
    print(S)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[0], n_fft = n_fft, hop_length = hop_length)
    ax[0].set(title= 'original spectrogram')
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1], n_fft=n_fft, hop_length=hop_length)
    new_times = librosa.times_like(f0, sr=sr, n_fft=n_fft, hop_length=hop_length)
    print("new_times: ", new_times.shape)
    print(new_times)
    # ax[1].set(title='pyin f0')
    # ax[1].plot(new_times, f0, label='fir_f0', color='cyan', linewidth=1)
    ax[1].set(title='pyin f0')
    ax[1].plot(new_times, f0, label='fir_f0', color='cyan', linewidth=1)
    ax[1].legend(loc='upper right')
    ax[1].label_outer()
    plt.show()

def extract_rms(n_fft, sr, hop_length, stft):
    S, phase = librosa.magphase(stft)
    rms = librosa.feature.rms(S = stft)
    fig, ax = plt.subplots(nrows=2, sharex=True)
    times = librosa.times_like(rms)
    ax[0].semilogy(times, rms[0], label='RMS Energy')
    ax[0].set(xticks=[])
    ax[0].legend(loc='upper right')
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')
    plt.show()

def extract_feature_two(dir):
    harmonic_ratio = []
    h_variance = []
    file_list = []
    for i in os.listdir(dir):
        file = os.path.splitext(i)
        if (file[1] != ".wav" and file[1] != ".mp3"):
            continue;
        y, sr = librosa.load(os.path.join(dir, i), sr = None)
        # y, sr = librosa.load(r"C:\Users\邢\Desktop\new_violin_audio", sr=None)
        n_fft = frame_length = 2048
        hop_length = int(n_fft/4)
        fre_sr = sr / n_fft
        print(y.shape)
        print(sr)
        #get timal feature
        tstft = librosa.stft(y, n_fft=n_fft, hop_length = hop_length)
        S , phase = librosa.magphase(tstft)
        rms = np.squeeze(librosa.feature.rms(S = S, frame_length = n_fft, hop_length = hop_length))

        f0, voiced_flag, voiced_probs = librosa.pyin(y, frame_length=n_fft, fmin= librosa.note_to_hz('G3'),
                                                     fmax=librosa.note_to_hz('C7'), sr=sr)
        print("f0: ", f0)
        # f0 = np.nan_to_num(f0)
        freq_num = get_num(f0.shape[0], f0, fre_sr)
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length = hop_length))
        print("stft:", stft.shape)
        harmonic = np.array([1, 2, 3, 4, 5, 6])
        energy_set, harmonic_set = get_energy(f0, stft, freq_num, harmonic)
        total_set = np.append(energy_set,harmonic_set,axis=1)
        print("energy:", energy_set.shape)
        # print("energy_ratio:", harmonic_set.shape)
        # print_set(harmonic_set)
        print("f0:", f0.shape)
        times = librosa.times_like(f0, sr=sr, n_fft=n_fft, hop_length=n_fft / 4)
        plt.plot(times, rms)
        plt.plot()

        D = librosa.amplitude_to_db(stft, ref=np.max)
        print("D:", D.shape)
        fig, ax = plt.subplots(2)
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax.flat[0], sr=sr, n_fft=n_fft, hop_length=n_fft / 4)
        ax.flat[0].set(title= i +'stft_original')
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        ax.flat[0].set(title= i + 'stft')
        ax.flat[0].legend(loc='upper right')
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax.flat[1], sr=sr, n_fft=n_fft,
                                       hop_length=n_fft / 4)
        ax.flat[1].plot(times, f0, label='f0', color='cyan', linewidth=1)
        ax.flat[1].plot(times, 2*f0, label='f1', color='g', linewidth=1)
        ax.flat[1].plot(times, 3*f0, label='f2', color='w', linewidth=1)
        ax.flat[1].plot(times, 4*f0, label='f3', color='g', linewidth=1)
        # ax.flat[1].plot(times, 5*f0, label='f4', color='cyan', linewidth=1)
        # ax.flat[1].plot(times, 6*f0, label='f5', color='w', linewidth=1)
        # ax.flat[1].plot(times, 7*f0, label='f6', color='cyan', linewidth=2)
        ax.flat[1].legend(loc='upper right')
        reduced_set = reduce_outliers(total_set,2)
        print_set(reduced_set[:, 2: 7])
        avg_set = np.mean(reduced_set[:, -5:], axis = 0)
        variance_set = np.var(reduced_set[:, 2: 7], axis = 0)
        h_variance.append(variance_set)
        print(h_variance)
        file_list.append(file[0])
        print(file_list)
        harmonic_ratio.append(avg_set)
        print(harmonic_ratio)
        # numpy.savetxt(file[0] + ".csv", reduced_set, delimiter= ',')
        plt.show()
    return np.array(file_list), np.array(harmonic_ratio), np.array(h_variance)

def extract_harmonic(y, sr, n_fft, hop_length):
    fre_sr = sr / n_fft
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length = hop_length))
    print("stft:", stft.shape)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, frame_length=n_fft, hop_length = hop_length, fmin= librosa.note_to_hz('G3'),
                                                 fmax=librosa.note_to_hz('C7'), sr=sr)
    avgPitch = np.mean(f0)
    devPitch = np.var(f0)
    # f0 = np.nan_to_num(f0)
    freq_num = get_num(f0.shape[0], f0, fre_sr)
    harmonic = np.array([1, 2, 3, 4, 5, 6])
    energy_set, harmonic_set = get_energy(f0, stft, freq_num, harmonic)
    print("energy:", energy_set.shape)
    # print("energy_ratio:", harmonic_set.shape)
    # print_set(harmonic_set)
    print("f0:", f0.shape)
    # reduced_set = reduce_outliers(total_set,2)
    # print_set(reduced_set[:, 2: 7])
    # avgHarmo_ratio = np.mean(harmonic_set, axis= 0)
    # dev_Harmo_ratio = np.var(harmonic_set, axis = 0)
    return f0, harmonic_set

if __name__ == '__main__':
    # dir = r"C:\Users\邢\Desktop\Recording_studio_audio"
    # dir2 = r"C:\Users\邢\Desktop\new_violin_audio\a1"
    # dir3 = r"C:\Users\邢\Desktop\new_violin_audio\a2"
    # # extract_feature(dir)
    # dirmaori = r"C:\Users\邢\Desktop\recordings\recordings"
    # dirnewest = r"C:\Users\邢\Desktop\2recording"
    # # extract_feature_two(dirmaori)
    # file_list, harmonic_ratio, h_variance = extract_feature_two(dirnewest)
    # fig, ax = plt.subplots()
    # ax.set(title="harmonic_ratio")
    # color_st = ['cyan', 'r', 'b', 'g']
    # color_index = 0;
    # for i in range(harmonic_ratio.shape[0]):
    #     ax.plot(range(1, 6, 1), harmonic_ratio[i], label=file_list[i], color=color_st[color_index], linewidth=1)
    #     plt.xticks(range(1,6,1))
    #     color_index = (color_index + 1) % 4
    # plt.legend(loc='upper right')
    #
    # fig1, ax1 = plt.subplots()
    # ax1.set(title="harmonic_variance")
    # color_index = 0;
    # for i in range(harmonic_ratio.shape[0]):
    #     ax1.plot(range(1, 6, 1), h_variance[i], label=file_list[i], color=color_st[color_index], linewidth=1)
    #     plt.xticks(range(1,6,1))
    #     color_index = (color_index + 1) % 4
    # plt.legend(loc='upper right')
    # plt.show()
    dir = r"C:\Users\xrw\Desktop\new_violin_audio"
    csv_path = []
    wav_path = []
    for i in os.listdir(dir):
        file = os.path.splitext(i)
        if (file[1] == ".wav" or file[1] == ".mp3"):
            wav_path.append(os.path.join(dir,file[0] + file[1]))
        if(file[1] == ".csv"):
            csv_path.append(os.path.join(dir,file[0] + file[1]))
    print(csv_path[1])
    print(wav_path)
    for i in range(4):
        show_multi_f0(csv_path[i], wav_path[i])
        # show_ori(wav_path[i])
    # show_multi_f0(csv_path1, wav_path1)
    # show_multi_f0(csv_path2, wav_path2)


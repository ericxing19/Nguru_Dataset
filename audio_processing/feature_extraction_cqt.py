import math

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy
import numpy as np
import scipy
import scipy.stats


def extract_rms(n_fft, sr, hop_length, stft):
    S, phase = librosa.magphase(stft)
    rms = librosa.feature.rms(S = S, frame_length = n_fft, hop_length = hop_length)
    print(rms.shape)
    avgrms = np.mean(rms)
    devrms = np.var(rms)
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # times = librosa.times_like(rms, sr = sr)
    # ax[0].semilogy(times, rms[0], label='RMS Energy')
    # ax[0].set(xticks=[])
    # ax[0].legend()
    # ax[0].label_outer()
    # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    #                          y_axis='log', x_axis='time', ax=ax[1],sr = sr)
    # ax[1].set(title='log Power spectrogram')
    # plt.show()
    return avgrms,devrms

def e_mfcc(mel, sr):
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), sr = sr, n_mfcc = 40)
    print("mfcc: ", mfcc.shape)
    avgmfcc = np.mean(mfcc, axis= 1)
    devmfcc = np.var(mfcc, axis= 1)
    print("avg mfcc: ", avgmfcc.shape)
    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # img = librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
    #                                x_axis='time', y_axis='mel', fmax=8000,
    #                                ax=ax[0],sr = sr)
    # fig.colorbar(img, ax=[ax[0]])
    # ax[0].set(title='Mel spectrogram')
    # ax[0].label_outer()
    # img = librosa.display.specshow(mfcc, x_axis='time', ax=ax[1])
    # fig.colorbar(img, ax=[ax[1]])
    # ax[1].set(title='MFCC')
    # plt.show()
    return avgmfcc,devmfcc

def e_zero_crossing_rate(n_fft, sr, hop_length, y):
    zcr = librosa.feature.zero_crossing_rate(y, frame_length = n_fft, hop_length = hop_length)
    avgzcr = np.mean(zcr)
    devzcr = np.mean(zcr)
    print("zcr shape: ", zcr.shape)
    return avgzcr,devzcr


def e_spectral_centroid(n_fft, sr, hop_length, y):
    S, phase = librosa.magphase(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    spec_cen = librosa.feature.spectral_centroid(y = y, n_fft = n_fft, hop_length = hop_length, sr = sr)
    print("spec_cen: ", spec_cen.shape)
    times = librosa.times_like(spec_cen, sr = sr)
    # frame = librosa.util.frame(y, frame_length=n_fft, hop_length=hop_length)
    S_abs = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
    # spectral spread
    ss = np.zeros((1, times.shape[0]))
    for i in range(times.shape[0]):
        ss[:, i] = np.sqrt((sum((i*sr/n_fft - spec_cen[:, i]) ** 2 * S_abs[:, i])) / sum(S_abs[:, i]))
    sk = np.zeros((1, times.shape[0]))
    for i in range(times.shape[0]):
        sk[:, i] = sum((i*sr/n_fft - spec_cen[:, i]) ** 4 * S_abs[:, i]) / (ss[:, i] ** 4 * sum(S_abs[:, i]))
    sc = np.zeros((1, times.shape[0]))
    for i in range(times.shape[0]):
        sc[:, i] = sum((i * sr / n_fft - spec_cen[:, i]) ** 3 * S_abs[:, i]) / (ss[:, i] ** 3 * sum(S_abs[:, i]))
    print("spec_spread: ", ss.shape)
    avgSpec_cen = numpy.mean(spec_cen)
    devSpec_cen = numpy.var(spec_cen)
    avgSpec_spread = np.mean(ss)
    devSpec_spread = np.var(ss)
    # fig, ax = plt.subplots()
    # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    #                          y_axis='log', x_axis='time', ax=ax, sr = sr)
    # ax.plot(times, spec_cen.T, label='Spectral centroid', color='w')
    # ax.legend(loc='upper right')
    # ax.set(title='log Power spectrogram')
    # plt.show()
    return avgSpec_cen, devSpec_cen, avgSpec_spread, devSpec_spread

def e_spectral_flatness(n_fft, sr, hop_length, S):
    flatness = librosa.feature.spectral_flatness(n_fft = n_fft, hop_length = hop_length, S = S)
    avgflat = numpy.mean(flatness)
    devflat = numpy.var(flatness)
    print("flatness: ", flatness.shape)
    return avgflat, devflat

def roll_off(n_fft, sr, hop_length, S):
    rolloff = librosa.feature.spectral_rolloff(n_fft = n_fft, sr = sr, hop_length = hop_length, S = S, roll_percent = 0.90)
    print("roll_off: ", rolloff.shape)
    avgrolloff = numpy.mean(rolloff)
    devrolloff = numpy.var(rolloff)
    # fig, ax = plt.subplots()
    # librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    #                          y_axis='log', x_axis='time', ax=ax, sr = sr)
    # ax.plot(librosa.times_like(rolloff,sr = sr), rolloff[0], label='Roll-off frequency (0.90)')
    # ax.legend(loc='lower right')
    # ax.set(title='log Power spectrogram')
    # plt.show()
    return avgrolloff, devrolloff

def get_num(length,f0, fre_sr):
    num_freq = np.zeros(length,dtype= int)
    for i in range(length):
        if (np.isnan(f0[i])):
            pass
        else:
            num_freq[i] = int(round(f0[i] / fre_sr));
    return num_freq

def get_energy(f0, S, num, harmonic):
    threshold = 2;
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

def extract_harmonic_f1(y, sr, n_fft, hop_length,rms):
    fre_sr = sr / n_fft

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length = hop_length))
    print("stft:", stft.shape)
    fk = np.linspace(0, sr, n_fft)
    fk = fk[0:(len(fk) // 2) + 1]
    S_abs = np.abs(stft)
    fk_matrix = np.transpose(np.tile(fk, (S_abs.shape[1], 1)))

    f0, voiced_flag, voiced_probs = librosa.pyin(y, frame_length=n_fft, hop_length = hop_length, fmin= librosa.note_to_hz('G3'),
                                                 fmax=librosa.note_to_hz('C7'), sr=sr)
    freq_num = get_num(f0.shape[0], f0, fre_sr)
    harmonic = np.array([1, 2, 3, 4, 5, 6])
    energy_set, harmonic_set = get_energy(f0, stft, freq_num, harmonic)
    print("energy_set :", energy_set.shape)
    # print("energy_ratio:", harmonic_set.shape)
    # print_set(harmonic_set)
    print("f0:", f0.shape)
    # reduced_set = reduce_outliers(total_set,2)
    # print_set(reduced_set[:, 2: 7])
    return f0, harmonic_set

def get_peak(array, threshold):
    if(threshold == 0):
        return [np.argmax(array)], [np.max(array)]
    peak_list = []
    for i in range(array.shape[0]):
        if(i >=threshold and i <= array.shape[0] - threshold -1):
            if(array[i] >= numpy.max(array[(i-threshold):(i+threshold + 1)])):
                peak = i
                peak_list.append(peak)
    if(len(peak_list) == 0):
        print("NAN")
        peak_list = get_peak(array,threshold-1)
    return np.array(peak_list)

# final ene extraction
def get_peak_har(array, threshold, pitch_range,pitch_range2):
    # har = all harmonic candidate, peak list = all peak got
    if(threshold == 0):
        return [np.argmax(array)], []
    peak_list = []
    har = []
    for i in range(array.shape[0]):
        if(i >=pitch_range and i <= array.shape[0] - pitch_range -1):
            if(array[i] >= np.max(array[(i-threshold):(i+threshold + 1)])):
                peak = i
                peak_list.append(peak)
                mean = np.mean(array[(i - pitch_range):(i + pitch_range + 1)])
                mean2 = np.mean(array[(i - pitch_range2):(i + pitch_range2 + 1)])
                if (array[i] >= 3*mean or array[i] >= 3 * mean2):
                    har.append(i)
    return np.array(peak_list), np.array(har)


def find_harmonic_ene(S_array, h_index, harmonic, threshold, special_idx):
    f0_index = h_index[0]
    # s_range = np.arange(h_index - threshold, h_index + threshold, 1)
    # print("f0 index: ", f0_index)
    # print("S_array: ", S_array)
    peak_list, har_candidate = get_peak_har(S_array,1,int(5 * h_index[0]/6), 20)
    total_ene = np.sum(S_array)
    ene_array = np.full([harmonic.shape[0]+1,], np.nan)
    pitch_ratio = np.full([harmonic.shape[0]+1,], np.nan)
    indx_array = np.zeros([harmonic.shape[0]+1,], dtype=int)
    for i in range(harmonic.shape[0]):
        min = 1000
        min_indx = 0
        for j in range(har_candidate.shape[0]):
            length = np.abs(har_candidate[j] - h_index[i])
            if (length < min):
                min = length
                min_indx = har_candidate[j]
        if(min < threshold):
           ene_array[i] = S_array[min_indx]
           indx_array[i] = min_indx

    # supplement the not detected harmonic
    for i in range(ene_array.shape[0] - 1):
        if (np.isnan(ene_array[i])):
            print(i+1, " harmonic detected err")
            min = 1000
            min_indx = 0
            for j in range(peak_list.shape[0]):
                length = np.abs(peak_list[j] - h_index[i])
                if (length < min):
                    min = length
                    min_indx = peak_list[j]
            if(length <= threshold):
                ene_array[i] = S_array[min_indx]
                indx_array[i] = min_indx
            else:
                ene_array[i] = np.max(S_array[(h_index[i] - 3): (h_index[i] + 3)])
                indx_array[i] = np.argmax(S_array[(h_index[i] - 3): (h_index[i] + 3)]) + h_index[i] - 3

    # new_set is special harmonic list
    new_set = [i for i in har_candidate if i in special_idx]
    # special harmonic list except other harmonic(1,2,3,4,5)
    n_set = [i for i in new_set if i not in indx_array]
    # get special harmonic
    # supplement the not detected special harmonic
    if(len(n_set) == 0):
        print("sp harmonic not detected")
        sp_h_set = [i for i in peak_list if i in special_idx]
        # special harmonic candidate list except other harmonic(1,2,3,4,5)
        sp_hn_set = [i for i in sp_h_set if i not in indx_array]
        if(len(sp_hn_set) == 0):
            print("sp harmonic NAN")
        else:
            ene_array[-1] = np.max(S_array[sp_hn_set])
            indx_array[-1] = sp_hn_set[np.argmax(S_array[sp_hn_set])]
    # detected
    else:
        max_s = 0
        max_indx = 0
        for ele in n_set:
            if (S_array[ele] > max_s):
                max_s = S_array[ele]
                max_indx = ele
        ene_array[-1] = max_s
        indx_array[-1] = max_indx
    fk = librosa.cqt_frequencies(360, fmin=librosa.note_to_hz('C4'),bins_per_octave=60)
    for i in range(indx_array.shape[0]):
        if(indx_array[i] == 0):
            pitch_ratio[i] = np.nan
        else:
            pitch_ratio[i] = fk[indx_array[i]] / fk[indx_array[0]]
    energy_array = ene_array
    ene_ratio = np.array(energy_array/total_ene)

    # x_range = np.arange(S_array.shape[0])
    # total_amp = np.sum(S_array)
    # amp_ratio = np.array(ene_array/total_amp)
    # pass_f0 = f0_index + 20
    # pass_f0 = 0
    # plt.title(str(f0_index) + "extraction")
    # plt.plot(x_range[pass_f0:] - pass_f0, S_array[pass_f0:])
    # color_set = ['r', 'b', 'g', 'k', 'y', 'c']
    # for h in range(harmonic.shape[0] + 1):
    #     plt.axvline(indx_array[h]-pass_f0, ymin=0, ymax= ene_array[h], color = color_set[h], label = str(h) + " harmonic" + str(amp_ratio[h]))
    # plt.legend()
    # plt.show()
    return energy_array, pitch_ratio, ene_ratio

def get_energy_f(f0, S, index, harmonic, special_idx):
    threshold = 6
    energy_array = np.zeros((S.shape[1], harmonic.shape[0] + 1))
    energy_ratio = np.zeros((S.shape[1],harmonic.shape[0] + 1))
    pitch_ratio = np.zeros((S.shape[1],harmonic.shape[0] + 1))
    for i in range(S.shape[1]):
        if (index[i][0] == 0):
            energy_array[i] = np.full([harmonic.shape[0] + 1,], np.nan)
            pitch_ratio[i] = np.full([harmonic.shape[0] + 1,], np.nan)
            energy_ratio[i] = np.full([harmonic.shape[0] + 1,], np.nan)
        else:
            e_array, p_ratio, e_ratio = find_harmonic_ene(S[:,i], index[i], harmonic, threshold, special_idx)
            energy_array[i] = e_array
            pitch_ratio[i] = p_ratio
            energy_ratio[i] = e_ratio

    return energy_array, pitch_ratio, energy_ratio

def get_ratio(freq1,freq2):
    a = freq1/freq2
    if(a < 1):
        a = 1/a
    return a


def extract_harmonic_f(y, sr, n_fft, hop_length):
    CQT = np.abs(librosa.cqt(y, sr, hop_length = hop_length, n_bins = 360, bins_per_octave=60, fmin=librosa.note_to_hz('C4')))
    fk = librosa.cqt_frequencies(360, fmin=librosa.note_to_hz('C4'),bins_per_octave=60)

    f0, voiced_flag, voiced_probs = librosa.pyin(y, frame_length=n_fft, hop_length = hop_length, fmin= librosa.note_to_hz('G3'),
                                                 fmax=librosa.note_to_hz('C7'), sr=sr)
    idx = 0
    fre_bin = CQT.shape[0]
    time_bin = CQT.shape[1]
    index = np.zeros((time_bin, 5), dtype=int)
    for i in range(CQT.shape[1]):
        if(np.isnan(f0[i])):
            continue
        else:
            for h in range(1, 6, 1):
                dis = 1000
                for j in range(CQT.shape[0]):
                    pitch_dif = get_ratio(f0[i]*h, fk[j])
                    if (pitch_dif < dis):
                        dis = pitch_dif
                        idx = j
                index[i,h-1] = int(idx)
    print(index)
    s_idx = 0
    s_dis = 10000
    s_idx2 = 0
    s_dis2 = 10000
    for j in range(CQT.shape[0]):
        s_pitch_dif = get_ratio(2200, fk[j])
        if (s_pitch_dif < s_dis):
            s_dis = s_pitch_dif
            s_idx = j
        s_pitch_dif2 = get_ratio(2650, fk[j])
        if(s_pitch_dif2 < s_dis2):
            s_dis2 = s_pitch_dif2
            s_idx2 = j
    special_idx = np.arange(int(s_idx),int(s_idx2),1)
    harmonic = np.array([1, 2, 3, 4, 5])
    log_power_CQT = np.log(CQT*CQT)
    power_CQT = CQT*CQT
    energy_set, pitch_ratio, energy_ratio = get_energy_f(f0, power_CQT, index, harmonic, special_idx)
    print("energy_set :", energy_set.shape)
    print(energy_set)
    print("pitch ratio: ", pitch_ratio)
    print("energy ratio: ", energy_ratio)
    # print("energy_ratio:", harmonic_set.shape)
    # print_set(harmonic_set)
    print("f0:", f0.shape)
    # reduced_set = reduce_outliers(total_set,2)
    # print_set(reduced_set[:, 2: 7])
    return f0, energy_set, pitch_ratio, energy_ratio

if __name__ == '__main__':
    y, sr = librosa.load(r"C:\Users\xrw\Desktop\interval_wav\n76800.wav",sr=None)
    # y, sr = librosa.load(r"C:\Users\邢\Desktop\interval_wav\n2029568.wav")
    n_fft = 2048
    hop_length = 512
    cqt = librosa.cqt(y, sr, hop_length = hop_length, n_bins = 360, bins_per_octave=60, fmin=librosa.note_to_hz('C4'))
    CQT = np.abs(cqt)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(CQT, ref=np.max),
                                   sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
    fig, ax = plt.subplots()
    fk = librosa.cqt_frequencies(360, fmin=librosa.note_to_hz('C4'), bins_per_octave=60)
    plt.plot(fk, CQT[:,1])
    pitch, energy_set, pitch_ratio, energy_ratio = extract_harmonic_f(y, sr, n_fft, hop_length)
    print(pitch_ratio)
    color_set = ['r', 'b', 'g', 'k', 'y', 'c']
    for h in range(CQT.shape[1]):
        if (h % 50 == 1):
            plt.figure()
            plt.title("harmonic")
            plt.yscale('log')
            plt.plot(fk, CQT[:,h])
            for j in range(6):
                if (j == 5):
                    print(energy_set[h, j])
                    plt.axvline(pitch[h] * pitch_ratio[h, j], ymin=0, ymax=0.03, color=color_set[j],
                                label="sp harmonic" + str(np.sqrt(energy_set[h, j])))
                else:
                    plt.axvline(pitch[h] * pitch_ratio[h, j], ymin=0, ymax=0.03, color=color_set[j],
                                label=str(j + 1) + " harmonic" + str(np.sqrt(energy_set[h, j])))
                    print(energy_set[h, j])
        plt.legend()

    # f0, energy_set, pitch_ratio, energy_ratio = extract_harmonic_f(y,sr,2048,512)
    plt.show()


import os

import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

import feature_extraction
import preprocessing
import csv

from newdataset.maori import feature_extraction_cqt


def generate_fft_bins(win_size, fs=44100):
    """
    Given the fft window size, generate the hz value per bin

    Parameters
    ----------
    win_size: window size in samples
    fs: sample rate
    Returns
    -------
    A vector containing the hz value per bin (fk)
    """
    # Generate win_size points from 0hz-fs
    fk = np.linspace(0, fs, win_size)

    # Remove symmetry and cut to fs/2
    return fk[0:(len(fk) // 2) + 1]

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

def extract_harmonic_m(dir):
    # instrument_list = []
    with open(r"C:\Users\邢\Desktop\10 nguru recording\totaldataset_withlabel.csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        title = ['avgPitch','devPitch', 'avgrms','devrms', 'avgzcr', 'devzcr', 'avgSpec_cen', 'devSpec_cen', 'avgSpec_spread', 'devSpec_spread', 'avgflat', 'devflat', 'avgrolloff', 'devrolloff', 'spec_kurt', 'spec_skew', 'avgharmonic_ratio2/1', 'avgharmonic_ratio3/1','avgharmonic_ratio4/1','avgharmonic_ratio5/1','avgharmonic_ratio6/1','devharmonic_ratio2/1','devharmonic_ratio3/1','devharmonic_ratio4/1','devharmonic_ratio5/1','devharmonic_ratio6/1']
        for i in range(0,40):
            print(i)
            title.extend(["avgmfcc"])
        for i in range(0,40):
            title.extend(["devmfcc"])
        w.writerow(title)
        dataset = np.array([])
        for i in os.listdir(dir):
            file = os.path.splitext(i)
            if (file[1] != ".wav" and file[1] != ".mp3"):
                continue;
            y, sr = librosa.load(os.path.join(dir, i), sr=None)
            n_fft = frame_length = 2048
            hop_length = int(n_fft / 4)
            intervals = preprocessing.preprecessing(y, sr)
            for ele in intervals:
                ys = y[ele[0]:ele[1]]
                sample = feature_extraction.get_All_Feature(y=ys, sr=sr, n_fft=n_fft, hop_length=hop_length)
                w.writerow(sample)
                np.append(dataset, sample, axis=0)
    print("data set: ", dataset.shape)
    print(len(dataset[0]))

def extract_harmonic(dir):
    with open(r"C:\Users\邢\Desktop\10 nguru recording\totaldataset_26withlabel.csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        title = ['avgPitch','devPitch', 'avgrms','devrms', 'avgzcr', 'devzcr', 'avgSpec_cen', 'devSpec_cen', 'avgSpec_spread', 'devSpec_spread', 'avgflat', 'devflat', 'avgrolloff', 'devrolloff', 'spec_kurt', 'spec_skew', 'avgharmonic_ratio2/1', 'avgharmonic_ratio3/1','avgharmonic_ratio4/1','avgharmonic_ratio5/1','avgharmonic_ratio6/1','devharmonic_ratio2/1','devharmonic_ratio3/1','devharmonic_ratio4/1','devharmonic_ratio5/1','devharmonic_ratio6/1','instrument']
        w.writerow(title)
        dataset = np.array([])
        instrument_num = 1
        for i in os.listdir(dir):
            file = os.path.splitext(i)
            if (file[1] != ".wav" and file[1] != ".mp3"):
                continue;
            y, sr = librosa.load(os.path.join(dir, i), sr=None)
            n_fft = frame_length = 2048
            hop_length = int(n_fft / 4)
            intervals, win_interval = preprocessing.preprecessing(y, sr, hop_length)
            for ele in intervals:
                ys = y[ele[0]:ele[1]]
                sample = feature_extraction.get_All_Feature(y=ys, sr=sr, n_fft=n_fft, hop_length=hop_length)
                sample = np.append(sample,instrument_num)
                w.writerow(sample)
                np.append(dataset, sample, axis=0)
            instrument_num += 1
    print("data set: ", dataset.shape)

#spcetral feature
def extract_All_feature(dir):
    with open(r"C:\Users\邢\Desktop\10 nguru recording\newfinalset.csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        title = ['avg_pitch', 'std_pitch', 'avg_rms', 'std_rms', 'avg_zcr', 'std_zcr', 'avg_spec_cen', 'std_spec_cen', 'avg_spectral_spread', 'std_spectral_spread', 'avg_spectral_centroid', 'std_spectral_centroid', 'avg_spectral_kurtosis', 'std_spectral_kurtosis', 'avg_flatness', 'std_flatness', 'avg_rolloff','std_rolloff','avg_f0_ene', 'std_f0_ene','avgharmonic_ratio2/1', 'avgharmonic_ratio3/1','avgharmonic_ratio4/1','avg_special_harmonic_ratio','stdharmonic_ratio2/1','stdharmonic_ratio3/1','stdharmonic_ratio4/1','std_special_harmonic_ratio','instrument']
        w.writerow(title)
        dataset = np.array([])
        instrument_num = 1
        for i in os.listdir(dir):
            file = os.path.splitext(i)
            if (file[1] != ".wav" and file[1] != ".mp3"):
                continue;
            y, sr = librosa.load(os.path.join(dir, i), sr=None)
            length = len(y)
            n_fft = frame_length = 2048
            hop_length = int(n_fft / 4)
            intervals, win_interval = preprocessing.preprecessing(y, sr, hop_length)
            S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, n_fft=n_fft,
                                                   hop_length=hop_length)
            stft = librosa.stft(y)
            S, phase = librosa.magphase(stft)
            S_abs = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
            pitch, energy_set, pitch_ratio, energy_ratio = feature_extraction.extract_harmonic_f(y, sr, n_fft, hop_length)
            print("pitch:", pitch.shape)
            print("total ene: ", energy_set.shape)
            print("total harmonic pitch ratio: ", pitch_ratio.shape)
            rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
            spec_cen = librosa.feature.spectral_centroid(y=y, n_fft=n_fft, hop_length=hop_length, sr=sr)
            times = librosa.times_like(spec_cen, sr=sr)
            ss = np.zeros((1, times.shape[0]))



            fk = np.linspace(0, sr, n_fft)
            fk = fk[0:(len(fk) // 2) + 1]
            fk_matrix = np.transpose(np.tile(fk, (S_abs.shape[1], 1)))
            print(fk_matrix.shape)
            print("fk_matrix: ", fk_matrix)
            for i in range(times.shape[0]):
                ss[:, i] = np.sqrt((sum((fk_matrix[:,i] - spec_cen[:, i]) ** 2 * S_abs[:, i])) / sum(S_abs[:, i]))
            print(ss)
            sk = np.zeros((1, times.shape[0]))
            for i in range(times.shape[0]):
                sk[:, i] = sum((fk_matrix[:,i] - spec_cen[:, i]) ** 4 * S_abs[:, i]) / (ss[:, i] ** 4 * sum(S_abs[:, i]))
            print(sk)
            sc = np.zeros((1, times.shape[0]))
            for i in range(times.shape[0]):
                sc[:, i] = sum((fk_matrix[:,i] - spec_cen[:, i]) ** 3 * S_abs[:, i]) / (ss[:, i] ** 3 * sum(S_abs[:, i]))
            flatness = librosa.feature.spectral_flatness(n_fft=n_fft, hop_length=hop_length, S=S)
            rolloff = librosa.feature.spectral_rolloff(n_fft=n_fft, sr=sr, hop_length=hop_length, S=S,
                                                       roll_percent=0.90)
            rms = np.squeeze(rms)
            zcr = np.squeeze(zcr)
            spec_cen = np.squeeze(spec_cen)
            ss = np.squeeze(ss)
            sk = np.squeeze(sk)
            sc = np.squeeze(sc)
            flatness = np.squeeze(flatness)
            rolloff = np.squeeze(rolloff)
            # pitch, harmonic_ratio = feature_extraction.extract_harmonic_f(y,sr, n_fft, hop_length)
            print("stft: ", stft.shape)
            print("rms: ", rms.shape)
            print("zcr: ", zcr.shape)
            print("spec_cen: ", spec_cen.shape)
            print("ss: ", ss.shape)
            print("sk: ", sk.shape)
            print("sc: ", sc.shape)
            print("time: ", times.shape)
            print("flatness: ", flatness.shape)
            print("rolloff: ", rolloff.shape)
            for i,ele in enumerate(win_interval):
                pitch_i = pitch[ele[0]:ele[1]]
                idx = np.where(~np.isnan(pitch_i))
                avg_pitch = np.mean(pitch_i[idx])
                std_pitch = np.std(pitch_i[idx])

                f0_energy_i = energy_set[ele[0]:ele[1], 0]
                avg_f0_energy = np.mean(f0_energy_i[idx])
                std_f0_energy = np.std(f0_energy_i[idx])
                harmonic_ratio_i = energy_ratio[ele[0]:ele[1]]
                avg_harmonic_ratio = np.mean(harmonic_ratio_i[idx], axis=0)
                std_harmonic_ratio = np.std(harmonic_ratio_i[idx], axis=0)

                avg_rms = np.mean(rms[ele[0]:ele[1]])
                std_rms = np.std(rms[ele[0]:ele[1]])
                avg_zcr = np.mean(zcr[ele[0]:ele[1]])
                std_zcr = np.std(zcr[ele[0]:ele[1]])
                avg_spec_cen = np.mean(spec_cen[ele[0]:ele[1]])
                std_spec_cen = np.std(spec_cen[ele[0]:ele[1]])
                avg_ss = np.mean(ss[ele[0]:ele[1]])
                std_ss = np.std(ss[ele[0]:ele[1]])
                avg_sk = np.mean(sk[ele[0]:ele[1]])
                std_sk = np.std(sk[ele[0]:ele[1]])
                avg_sc = np.mean(sc[ele[0]:ele[1]])
                std_sc = np.std(sc[ele[0]:ele[1]])
                avg_flatness = np.mean(flatness[ele[0]:ele[1]])
                std_flatness = np.std(flatness[ele[0]:ele[1]])
                avg_rolloff = np.mean(rolloff[ele[0]:ele[1]])
                std_rolloff = np.std(rolloff[ele[0]:ele[1]])
                sample = [avg_pitch, std_pitch, avg_rms, std_rms, avg_zcr, std_zcr, avg_spec_cen, std_spec_cen, avg_ss, std_ss, avg_sc, std_sc, avg_sk, std_sk, avg_flatness, std_flatness, avg_rolloff, std_rolloff, avg_f0_energy, std_f0_energy]
                sample.extend(avg_harmonic_ratio)
                sample.extend(std_harmonic_ratio)
                sample = np.append(sample,instrument_num)
                w.writerow(sample)
                np.append(dataset, sample, axis=0)
            instrument_num += 1
    print("data set: ", dataset.shape)


def extract_all():
    with open(r"C:\Users\邢\Desktop\10 nguru recording\newfinalset.csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        title = ['avg_pitch', 'std_pitch', 'avg_rms', 'std_rms', 'avg_zcr', 'std_zcr', 'avg_spec_cen', 'std_spec_cen', 'avg_spectral_spread', 'std_spectral_spread', 'avg_spectral_centroid', 'std_spectral_centroid', 'avg_spectral_kurtosis', 'std_spectral_kurtosis', 'avg_flatness', 'std_flatness', 'avg_rolloff','std_rolloff','avg_f0_ene', 'std_f0_ene','avgharmonic_ratio2/1', 'avgharmonic_ratio3/1','avgharmonic_ratio4/1','avg_special_harmonic_ratio','stdharmonic_ratio2/1','stdharmonic_ratio3/1','stdharmonic_ratio4/1','std_special_harmonic_ratio','instrument']
        w.writerow(title)
        dataset = np.array([])
        instrument_num = 1
        for i in os.listdir(dir):
            file = os.path.splitext(i)
            if (file[1] != ".wav" and file[1] != ".mp3"):
                continue;
            y, sr = librosa.load(os.path.join(dir, i), sr=None)
            length = len(y)
            n_fft = frame_length = 2048
            hop_length = int(n_fft / 4)
            intervals, win_interval = preprocessing.preprecessing(y, sr, hop_length)
            S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, n_fft=n_fft,
                                                   hop_length=hop_length)
            stft = librosa.stft(y)
            S, phase = librosa.magphase(stft)
            S_abs = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
            pitch, energy_set, pitch_ratio, energy_ratio = feature_extraction.extract_harmonic_f(y, sr, n_fft, hop_length)
            print("pitch:", pitch.shape)
            print("total ene: ", energy_set.shape)
            print("total harmonic pitch ratio: ", pitch_ratio.shape)
            rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
            spec_cen = librosa.feature.spectral_centroid(y=y, n_fft=n_fft, hop_length=hop_length, sr=sr)
            times = librosa.times_like(spec_cen, sr=sr)
            ss = np.zeros((1, times.shape[0]))



            fk = np.linspace(0, sr, n_fft)
            fk = fk[0:(len(fk) // 2) + 1]
            fk_matrix = np.transpose(np.tile(fk, (S_abs.shape[1], 1)))
            print(fk_matrix.shape)
            print("fk_matrix: ", fk_matrix)
            for i in range(times.shape[0]):
                ss[:, i] = np.sqrt((sum((fk_matrix[:,i] - spec_cen[:, i]) ** 2 * S_abs[:, i])) / sum(S_abs[:, i]))
            print(ss)
            sk = np.zeros((1, times.shape[0]))
            for i in range(times.shape[0]):
                sk[:, i] = sum((fk_matrix[:,i] - spec_cen[:, i]) ** 4 * S_abs[:, i]) / (ss[:, i] ** 4 * sum(S_abs[:, i]))
            print(sk)
            sc = np.zeros((1, times.shape[0]))
            for i in range(times.shape[0]):
                sc[:, i] = sum((fk_matrix[:,i] - spec_cen[:, i]) ** 3 * S_abs[:, i]) / (ss[:, i] ** 3 * sum(S_abs[:, i]))
            flatness = librosa.feature.spectral_flatness(n_fft=n_fft, hop_length=hop_length, S=S)
            rolloff = librosa.feature.spectral_rolloff(n_fft=n_fft, sr=sr, hop_length=hop_length, S=S,
                                                       roll_percent=0.90)
            rms = np.squeeze(rms)
            zcr = np.squeeze(zcr)
            spec_cen = np.squeeze(spec_cen)
            ss = np.squeeze(ss)
            sk = np.squeeze(sk)
            sc = np.squeeze(sc)
            flatness = np.squeeze(flatness)
            rolloff = np.squeeze(rolloff)
            # pitch, harmonic_ratio = feature_extraction.extract_harmonic_f(y,sr, n_fft, hop_length)
            print("stft: ", stft.shape)
            print("rms: ", rms.shape)
            print("zcr: ", zcr.shape)
            print("spec_cen: ", spec_cen.shape)
            print("ss: ", ss.shape)
            print("sk: ", sk.shape)
            print("sc: ", sc.shape)
            print("time: ", times.shape)
            print("flatness: ", flatness.shape)
            print("rolloff: ", rolloff.shape)
            for i,ele in enumerate(win_interval):
                pitch_i = pitch[ele[0]:ele[1]]
                idx = np.where(~np.isnan(pitch_i))
                avg_pitch = np.mean(pitch_i[idx])
                std_pitch = np.std(pitch_i[idx])

                f0_energy_i = energy_set[ele[0]:ele[1], 0]
                avg_f0_energy = np.mean(f0_energy_i[idx])
                std_f0_energy = np.std(f0_energy_i[idx])
                harmonic_ratio_i = energy_ratio[ele[0]:ele[1]]
                avg_harmonic_ratio = np.mean(harmonic_ratio_i[idx], axis=0)
                std_harmonic_ratio = np.std(harmonic_ratio_i[idx], axis=0)

                avg_rms = np.mean(rms[ele[0]:ele[1]])
                std_rms = np.std(rms[ele[0]:ele[1]])
                avg_zcr = np.mean(zcr[ele[0]:ele[1]])
                std_zcr = np.std(zcr[ele[0]:ele[1]])
                avg_spec_cen = np.mean(spec_cen[ele[0]:ele[1]])
                std_spec_cen = np.std(spec_cen[ele[0]:ele[1]])
                avg_ss = np.mean(ss[ele[0]:ele[1]])
                std_ss = np.std(ss[ele[0]:ele[1]])
                avg_sk = np.mean(sk[ele[0]:ele[1]])
                std_sk = np.std(sk[ele[0]:ele[1]])
                avg_sc = np.mean(sc[ele[0]:ele[1]])
                std_sc = np.std(sc[ele[0]:ele[1]])
                avg_flatness = np.mean(flatness[ele[0]:ele[1]])
                std_flatness = np.std(flatness[ele[0]:ele[1]])
                avg_rolloff = np.mean(rolloff[ele[0]:ele[1]])
                std_rolloff = np.std(rolloff[ele[0]:ele[1]])
                sample = [avg_pitch, std_pitch, avg_rms, std_rms, avg_zcr, std_zcr, avg_spec_cen, std_spec_cen, avg_ss, std_ss, avg_sc, std_sc, avg_sk, std_sk, avg_flatness, std_flatness, avg_rolloff, std_rolloff, avg_f0_energy, std_f0_energy]
                sample.extend(avg_harmonic_ratio)
                sample.extend(std_harmonic_ratio)
                sample = np.append(sample,instrument_num)
                w.writerow(sample)
                np.append(dataset, sample, axis=0)
            instrument_num += 1
    print("data set: ", dataset.shape)

def new_feature(dir, if_dct, method, write_path):
    with open(write_path, 'w', newline='') as cfile:
        w = csv.writer(cfile)
        title = ['avg_zcr', 'total_energy', 'attack_energy', 'energy1','energy2', 'energy3','energy4','energy5']
        title.extend(['pitch ratio1/1', 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio4/1', 'pitch ratio5/1', 'pitch ratio sp/1'])
        title.extend(['harmonic energy ratio1', 'harmonic energy ratio2', 'harmonic energy ratio3', 'harmonic energy ratio4', 'harmonic energy ratio5', 'harmonic energy ratio sp'])
        title.extend(['attack pitch ratio1/1', 'attack pitch ratio2/1', 'attack pitch ratio3/1', 'attack pitch ratio4/1', 'attack pitch ratio5/1', 'attack pitch ratio sp/1',])
        title.extend(['att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'att harmonic ratio4', 'att harmonic ratio5', 'att harmonic ratio sp'])
        title.extend(['1st harmonic1', '2nd harmonic1', '3rd harmonic1', '4th harmonic1', '5th harmonic1'])
        title.extend(['1st harmonic2', '2nd harmonic2', '3rd harmonic2', '4th harmonic2', '5th harmonic2'])
        title.extend(['1st harmonic3', '2nd harmonic3', '3rd harmonic3', '4th harmonic3', '5th harmonic3'])
        title.extend(['1st harmonic4', '2nd harmonic4', '3rd harmonic4', '4th harmonic4', '5th harmonic4'])
        title.extend(['1st harmonic5', '2nd harmonic5', '3rd harmonic5', '4th harmonic5', '5th harmonic5'])
        title.extend(['1st harmonicsp', '2nd harmonicsp', '3rd harmonicsp', '4th harmonicsp', '5th harmonicsp'])
        title.extend(['instrument'])

        w.writerow(title)
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

            n_fft = frame_length = 4096
            hop_length = int(n_fft / 4)
            final_interval, win_interval, attack, five_part = preprocessing.preprecessing1(y, sr, n_fft = 1024, top_db = 40, output_hop_length = hop_length)

            S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, n_fft=n_fft,
                                                   hop_length=hop_length)
            stft = librosa.stft(y, n_fft = n_fft, hop_length = hop_length)
            S, phase = librosa.magphase(stft)
            S_abs = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
            if(if_dct):
                pitch, energy_set, pitch_ratio, energy_ratio = feature_extraction_cqt.extract_harmonic_f(y, sr, n_fft, hop_length)
            else:
                pitch, energy_set, pitch_ratio, energy_ratio = feature_extraction.extract_harmonic_f(y, sr, n_fft,
                                                                                                     hop_length)
            print("pitch:", pitch.shape)
            print("total ene: ", energy_set.shape)
            print("total harmonic pitch ratio: ", pitch_ratio.shape)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
            zcr = np.squeeze(zcr)

            print("stft: ", stft.shape)
            print("zcr: ", zcr.shape)
            for i,ele in enumerate(win_interval):
                # ele[1] += 1
                print(ele)
                # Get index
                # total part
                pitch_i = pitch[ele[0]:ele[1]]
                idx = np.where(~np.isnan(pitch_i))
                # attack part
                attack_ele = attack[i] + 1
                print("attack_ele ", attack_ele)
                pitch_i = pitch[ele[0]:attack_ele]
                a_idx = np.where(~np.isnan(pitch_i))
                # five part
                five_part_ele = five_part[i]
                pitch_1 = pitch[five_part_ele[0]:five_part_ele[1]]
                idx1 = np.where(~np.isnan(pitch_1))
                pitch_2 = pitch[five_part_ele[1]:five_part_ele[2]]
                idx2 = np.where(~np.isnan(pitch_2))
                pitch_3 = pitch[five_part_ele[2]:five_part_ele[3]]
                idx3 = np.where(~np.isnan(pitch_3))
                pitch_4 = pitch[five_part_ele[3]:five_part_ele[4]]
                idx4 = np.where(~np.isnan(pitch_4))
                pitch_5 = pitch[five_part_ele[4]:five_part_ele[5]]
                idx5 = np.where(~np.isnan(pitch_5))

                if(method == 'mean'):
                    # total part
                    # avg_pitch = np.mean(pitch_i[idx])
                    # std_pitch = np.std(pitch_i[idx])

                    total_energy = np.nanmean(energy_set[ele[0]:ele[1]][idx])

                    harmonic_ratio_i = energy_ratio[ele[0]:ele[1]]
                    avg_harmonic_ratio = np.nanmean(harmonic_ratio_i[idx], axis=0)
                    # std_harmonic_ratio = np.std(harmonic_ratio_i[idx], axis=0)
                    pitch_ratio_i = pitch_ratio[ele[0]:ele[1]]
                    avg_pitch_ratio = np.nanmean(pitch_ratio_i[idx], axis=0)

                    avg_zcr = np.nanmean(zcr[ele[0]:ele[1]][idx])
                    std_zcr = np.std(zcr[ele[0]:ele[1]])

                    # attack part
                    attack_energy = np.nanmean(energy_set[ele[0]:attack_ele][a_idx])

                    attack_harmonic_ratio_i = energy_ratio[ele[0]:attack_ele]
                    attack_harmonic_ratio = np.nanmean(attack_harmonic_ratio_i[a_idx], axis=0)
                    # std_harmonic_ratio = np.std(attack_harmonic_ratio_i[idx], axis=0)

                    attack_pitch_ratio_i = pitch_ratio[ele[0]:attack_ele]
                    attack_pitch_ratio = np.nanmean(attack_pitch_ratio_i[a_idx], axis=0)

                    # five part
                    energy1 = np.nanmean(energy_set[five_part_ele[0]:five_part_ele[1]][idx1])
                    energy2 = np.nanmean(energy_set[five_part_ele[1]:five_part_ele[2]][idx2])
                    energy3 = np.nanmean(energy_set[five_part_ele[2]:five_part_ele[3]][idx3])
                    energy4 = np.nanmean(energy_set[five_part_ele[3]:five_part_ele[4]][idx4])
                    energy5 = np.nanmean(energy_set[five_part_ele[4]:five_part_ele[5]][idx5])
                    # harmonic_ratio_1 represents part 1's [1,2,3,4,5,sp]
                    harmonic_ratio_1 = np.nanmean(energy_ratio[five_part_ele[0]:five_part_ele[1]][idx1], axis=0)
                    harmonic_ratio_2 = np.nanmean(energy_ratio[five_part_ele[1]:five_part_ele[2]][idx2], axis=0)
                    harmonic_ratio_3 = np.nanmean(energy_ratio[five_part_ele[2]:five_part_ele[3]][idx3], axis=0)
                    harmonic_ratio_4 = np.nanmean(energy_ratio[five_part_ele[3]:five_part_ele[4]][idx4], axis=0)
                    harmonic_ratio_5 = np.nanmean(energy_ratio[five_part_ele[4]:five_part_ele[5]][idx5], axis=0)
                elif (method == 'median'):
                    # total part
                    # avg_pitch = np.median(pitch_i[idx])

                    total_energy = np.nanmedian(energy_set[ele[0]:ele[1]][idx])

                    harmonic_ratio_i = energy_ratio[ele[0]:ele[1]]
                    avg_harmonic_ratio = np.nanmedian(harmonic_ratio_i[idx], axis=0)
                    # std_harmonic_ratio = np.std(harmonic_ratio_i[idx], axis=0)
                    pitch_ratio_i = pitch_ratio[ele[0]:ele[1]]
                    avg_pitch_ratio = np.nanmedian(pitch_ratio_i[idx], axis=0)

                    avg_zcr = np.nanmedian(zcr[ele[0]:ele[1]][idx])
                    std_zcr = np.std(zcr[ele[0]:ele[1]])

                    # attack part
                    attack_energy = np.nanmedian(energy_set[ele[0]:attack_ele][a_idx])

                    attack_harmonic_ratio_i = energy_ratio[ele[0]:attack_ele]
                    attack_harmonic_ratio = np.nanmedian(attack_harmonic_ratio_i[a_idx], axis=0)
                    # std_harmonic_ratio = np.std(attack_harmonic_ratio_i[idx], axis=0)

                    attack_pitch_ratio_i = pitch_ratio[ele[0]:attack_ele]
                    attack_pitch_ratio = np.nanmedian(attack_pitch_ratio_i[a_idx], axis=0)

                    # five part
                    energy1 = np.nanmedian(energy_set[five_part_ele[0]:five_part_ele[1]][idx1])
                    energy2 = np.nanmedian(energy_set[five_part_ele[1]:five_part_ele[2]][idx2])
                    energy3 = np.nanmedian(energy_set[five_part_ele[2]:five_part_ele[3]][idx3])
                    energy4 = np.nanmedian(energy_set[five_part_ele[3]:five_part_ele[4]][idx4])
                    energy5 = np.nanmedian(energy_set[five_part_ele[4]:five_part_ele[5]][idx5])

                    harmonic_ratio_1 = np.nanmedian(energy_ratio[five_part_ele[0]:five_part_ele[1]][idx1], axis=0)
                    harmonic_ratio_2 = np.nanmedian(energy_ratio[five_part_ele[1]:five_part_ele[2]][idx2], axis=0)
                    harmonic_ratio_3 = np.nanmedian(energy_ratio[five_part_ele[2]:five_part_ele[3]][idx3], axis=0)
                    harmonic_ratio_4 = np.nanmedian(energy_ratio[five_part_ele[3]:five_part_ele[4]][idx4], axis=0)
                    harmonic_ratio_5 = np.nanmedian(energy_ratio[five_part_ele[4]:five_part_ele[5]][idx5], axis=0)

                # five_part_harmonic_ratio.extend(harmonic_ratio_1)
                # five_part_harmonic_ratio.extend(harmonic_ratio_2)
                # five_part_harmonic_ratio.extend(harmonic_ratio_3)
                # five_part_harmonic_ratio.extend(harmonic_ratio_4)
                # five_part_harmonic_ratio.extend(harmonic_ratio_5)
                five_part_harmonic_ratio = []
                for i in range(harmonic_ratio_1.shape[0]):
                    five_part_harmonic_ratio.append(harmonic_ratio_1[i])
                    five_part_harmonic_ratio.append(harmonic_ratio_2[i])
                    five_part_harmonic_ratio.append(harmonic_ratio_3[i])
                    five_part_harmonic_ratio.append(harmonic_ratio_4[i])
                    five_part_harmonic_ratio.append(harmonic_ratio_5[i])
                # five_part_harmonic_ratio = np.array(five_part_harmonic_ratio)

                sample = [avg_zcr, total_energy, attack_energy, energy1,energy2, energy3,energy4,energy5]
                sample.extend(avg_pitch_ratio)
                sample.extend(avg_harmonic_ratio)
                sample.extend(attack_pitch_ratio)
                sample.extend(attack_harmonic_ratio)
                sample.extend(five_part_harmonic_ratio)
                sample = np.append(sample,instrument_num)
                print(sample.shape)
                w.writerow(sample)
                np.append(dataset, sample, axis=0)
            instrument_num += 1
    print("data set: ", dataset.shape)

def test_e(dir):
    with open(r"C:\Users\邢\Desktop\10 nguru recording\2022_11_25.csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        title = ['avg_zcr', 'total_energy', 'attack_energy', 'energy1','energy2', 'energy3','energy4','energy5']
        title.extend(['pitch ratio1/1', 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio4/1', 'pitch ratio5/1', 'pitch ratio sp/1'])
        title.extend(['harmonic energy ratio1', 'harmonic energy ratio2', 'harmonic energy ratio3', 'harmonic energy ratio4', 'harmonic energy ratio5', 'harmonic energy ratio sp'])
        title.extend(['attack pitch ratio1/1', 'attack pitch ratio2/1', 'attack pitch ratio3/1', 'attack pitch ratio4/1', 'attack pitch ratio5/1', 'attack pitch ratio sp/1',])
        title.extend(['att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'att harmonic ratio4', 'att harmonic ratio5', 'att harmonic ratio sp'])
        title.extend(['1st harmonic1', '2nd harmonic1', '3rd harmonic1', '4th harmonic1', '5th harmonic1'])
        title.extend(['1st harmonic2', '2nd harmonic2', '3rd harmonic2', '4th harmonic2', '5th harmonic2'])
        title.extend(['1st harmonic3', '2nd harmonic3', '3rd harmonic3', '4th harmonic3', '5th harmonic3'])
        title.extend(['1st harmonic4', '2nd harmonic4', '3rd harmonic4', '4th harmonic4', '5th harmonic4'])
        title.extend(['1st harmonic5', '2nd harmonic5', '3rd harmonic5', '4th harmonic5', '5th harmonic5'])
        title.extend(['1st harmonicsp', '2nd harmonicsp', '3rd harmonicsp', '4th harmonicsp', '5th harmonicsp'])
        title.extend(['instrument'])


        w.writerow(title)
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
            if(file[0] != "nguru-1"):
                continue
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

            # S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, n_fft=n_fft,
            #                                        hop_length=hop_length)
            stft = librosa.stft(y, n_fft = n_fft, hop_length = hop_length)
            S, phase = librosa.magphase(stft)
            # S_abs = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
            pitch = np.load("extraction_result/Zpitch.npy")
            pitch_ratio = np.load("extraction_result/Zpitch_ratio.npy")
            energy_ratio = np.load("extraction_result/Zenergy_ratio.npy")
            energy_set = np.load("extraction_result/Zene_set.npy")

            # pitch, energy_set, pitch_ratio, energy_ratio = feature_extraction_cqt.extract_harmonic_f(y, sr, n_fft, hop_length)
            # print("pitch:", pitch.shape)
            # print("total ene: ", energy_set.shape)
            # print("total harmonic pitch ratio: ", pitch_ratio.shape)
            # np.save('1CQTpitch', pitch)
            # np.save('1CQTene_set', energy_set)
            # np.save("1CQTpitch_ratio", pitch_ratio)
            # np.save('1CQTenergy_ratio', energy_ratio)
            # for ele in pitch_ratio:
            #     print(ele)
            for i,ele in enumerate(win_interval):
                if(i >= 61 or i <=59):
                    continue
                fig, ax = plt.subplots()
                librosa.display.specshow(librosa.amplitude_to_db(S[:, ele[0]:ele[1]], ref=np.max), sr = sr,
                                         y_axis='cqt_note', x_axis='time', ax=ax, hop_length=hop_length)
                ax.set(title='original spectrogram')
                ele[1] += 1
                print(pitch[ele[0]:ele[1]])
                print(ele)
                for h in range(ele[0], ele[1], 1):
                    if(h % 20 != 1):
                        continue
                    print(h, str(i) + " pitch: ", pitch[h])
                    print(h, " energy: ", energy_set[h])
                    print(h, " energy ratio: ", energy_ratio[h])
                    print(h, " amp: ", np.sqrt(energy_set[h]))
                    S_array = S[:, h]
                    x_range = np.arange(S_array.shape[0])
                    pass_f0 = 0
                    color_set = ['r', 'b', 'g', 'k', 'y', 'c']

                    plt.figure()
                    plt.title("ori harmonic")
                    plt.plot(fk[pass_f0:350], S_array[pass_f0:350])

                    plt.figure()
                    plt.title("harmonic")
                    plt.plot(fk[pass_f0:350], S_array[pass_f0:350])
                    one_bin = fk[2]/fk[1]
                    for j in range(6):
                        if(j == 5):
                            plt.axvline(pitch[h] * pitch_ratio[h, j]/(one_bin**pass_f0), ymin=0, ymax=0.03, color=color_set[j],
                                        label="sp harmonic" + str(np.sqrt(energy_set[h, j])))
                        else:
                            plt.axvline(pitch[h] * pitch_ratio[h, j]/(one_bin**pass_f0), ymin=0, ymax=0.03, color=color_set[j],
                                        label=str(j + 1) + " harmonic" + str(np.sqrt(energy_set[h, j])))
                    plt.legend()

                    plt.figure()
                    pass_f0 = 100
                    plt.yticks(np.arange(0, max(S_array[pass_f0:]), 0.05))
                    plt.title("harmonic2")
                    plt.plot(fk[pass_f0:]- fk[pass_f0], S_array[pass_f0:])
                    for j in range(6):
                        if (j == 5):
                            plt.axvline(pitch[h] * pitch_ratio[h, j] - fk[pass_f0], ymin=0, ymax=0.03,
                                        color=color_set[j],
                                        label="sp harmonic" + str(np.sqrt(energy_set[h, j])))
                        else:
                            plt.axvline(pitch[h] * pitch_ratio[h, j] - fk[pass_f0], ymin=0, ymax=0.03, color=color_set[j],
                                        label=str(j+1) + " harmonic" + str(np.sqrt(energy_set[h, j])))
                    plt.legend()

                    plt.show()

    print("data set: ", dataset.shape)

def test_e_cqt(dir):
    with open(r"C:\Users\邢\Desktop\10 nguru recording\2022_11_25.csv", 'w', newline='') as cfile:
        w = csv.writer(cfile)
        title = ['avg_zcr', 'total_energy', 'attack_energy', 'energy1','energy2', 'energy3','energy4','energy5']
        title.extend(['pitch ratio1/1', 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio4/1', 'pitch ratio5/1', 'pitch ratio sp/1'])
        title.extend(['harmonic energy ratio1', 'harmonic energy ratio2', 'harmonic energy ratio3', 'harmonic energy ratio4', 'harmonic energy ratio5', 'harmonic energy ratio sp'])
        title.extend(['attack pitch ratio1/1', 'attack pitch ratio2/1', 'attack pitch ratio3/1', 'attack pitch ratio4/1', 'attack pitch ratio5/1', 'attack pitch ratio sp/1',])
        title.extend(['att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'att harmonic ratio4', 'att harmonic ratio5', 'att harmonic ratio sp'])
        title.extend(['1st harmonic1', '2nd harmonic1', '3rd harmonic1', '4th harmonic1', '5th harmonic1'])
        title.extend(['1st harmonic2', '2nd harmonic2', '3rd harmonic2', '4th harmonic2', '5th harmonic2'])
        title.extend(['1st harmonic3', '2nd harmonic3', '3rd harmonic3', '4th harmonic3', '5th harmonic3'])
        title.extend(['1st harmonic4', '2nd harmonic4', '3rd harmonic4', '4th harmonic4', '5th harmonic4'])
        title.extend(['1st harmonic5', '2nd harmonic5', '3rd harmonic5', '4th harmonic5', '5th harmonic5'])
        title.extend(['1st harmonicsp', '2nd harmonicsp', '3rd harmonicsp', '4th harmonicsp', '5th harmonicsp'])
        title.extend(['instrument'])

        w.writerow(title)
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
            if(file[0] != "nguru-1"):
                continue
            # if (file[1] != ".wav" and file[1] != ".mp3"):
            #     continue;
            print(file)
            y, sr = librosa.load(os.path.join(dir, i), sr=None)
            length = len(y)

            n_fft = frame_length = 2048
            hop_length = int(n_fft / 4)
            fk = np.linspace(0, sr, n_fft)
            fk = fk[0:(len(fk) // 2) + 1]
            fk = librosa.cqt_frequencies(360, fmin=librosa.note_to_hz('C4'), bins_per_octave=60)
            final_interval, win_interval, attack, five_part = preprocessing.preprecessing1(y, sr, 512, 40)

            # S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, n_fft=n_fft,
            #                                        hop_length=hop_length)
            # stft = librosa.stft(y, n_fft = n_fft, hop_length = hop_length)
            # S, phase = librosa.magphase(stft)
            cqt = librosa.cqt(y, sr, hop_length=hop_length, n_bins=360, bins_per_octave=60,
                              fmin=librosa.note_to_hz('C4'))
            CQT = np.abs(cqt)
            # S_abs = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length))
            # pitch = np.load("Zpitch.npy")
            # pitch_ratio = np.load("Zpitch_ratio.npy")
            # energy_ratio = np.load("Zenergy_ratio.npy")
            # energy_set = np.load("Zene_set.npy")
            pitch = np.load('extraction_result/1CQTpitch.npy')
            energy_set = np.load('extraction_result/1CQTene_set.npy')
            pitch_ratio = np.load("extraction_result/1CQTpitch_ratio.npy")
            energy_ratio = np.load('extraction_result/1CQTenergy_ratio.npy')
            # pitch, energy_set, pitch_ratio, energy_ratio = feature_extraction_cqt.extract_harmonic_f(y, sr, n_fft, hop_length)
            # print("pitch:", pitch.shape)
            # print("total ene: ", energy_set.shape)
            # print("total harmonic pitch ratio: ", pitch_ratio.shape)
            # np.save('1CQTpitch', pitch)
            # np.save('1CQTene_set', energy_set)
            # np.save("1CQTpitch_ratio", pitch_ratio)
            # np.save('1CQTenergy_ratio', energy_ratio)
            # for ele in pitch_ratio:
            #     print(ele)
            print("CQT shape:", CQT.shape)
            for i,ele in enumerate(win_interval):
                if(i >= 61 or i <=59):
                    continue
                fig, ax = plt.subplots()
                librosa.display.specshow(librosa.amplitude_to_db(CQT[:, ele[0]:ele[1]], ref=np.max), sr = sr,
                                         y_axis='cqt_note', x_axis='time', ax=ax, hop_length=hop_length)
                ax.set(title='original spectrogram')
                ele[1] += 1
                print(pitch[ele[0]:ele[1]])
                print(ele)
                for h in range(ele[0], ele[1], 1):
                    if(h % 20 != 1):
                        continue
                    print(h, str(i) + " pitch: ", pitch[h])
                    print(h, " energy: ", energy_set[h])
                    print(h, " energy ratio: ", energy_ratio[h])
                    print(h, " amp: ", np.sqrt(energy_set[h]))
                    S_array = CQT[:, h]
                    x_range = np.arange(S_array.shape[0])
                    pass_f0 = 0
                    color_set = ['r', 'b', 'g', 'k', 'y', 'c']

                    plt.figure()
                    plt.title("ori harmonic")
                    plt.plot(fk[pass_f0:], S_array[pass_f0:])

                    plt.figure()
                    plt.title("harmonic")
                    plt.plot(fk[pass_f0:], S_array[pass_f0:])
                    one_bin = fk[2]/fk[1]
                    for j in range(6):
                        if(j == 5):
                            plt.axvline(pitch[h] * pitch_ratio[h, j]/(one_bin**pass_f0), ymin=0, ymax=0.03, color=color_set[j],
                                        label="sp harmonic" + str(np.sqrt(energy_set[h, j])))
                        else:
                            plt.axvline(pitch[h] * pitch_ratio[h, j]/(one_bin**pass_f0), ymin=0, ymax=0.03, color=color_set[j],
                                        label=str(j + 1) + " harmonic" + str(np.sqrt(energy_set[h, j])))
                    plt.legend()

                    plt.figure()
                    pass_f0 = 100
                    plt.yticks(np.arange(0, max(S_array[pass_f0:]), 0.05))
                    plt.title("harmonic2")
                    plt.plot(fk[pass_f0:]- fk[pass_f0], S_array[pass_f0:])
                    for j in range(6):
                        if (j == 5):
                            plt.axvline(pitch[h] * pitch_ratio[h, j] - fk[pass_f0], ymin=0, ymax=0.03,
                                        color=color_set[j],
                                        label="sp harmonic" + str(np.sqrt(energy_set[h, j])))
                        else:
                            plt.axvline(pitch[h] * pitch_ratio[h, j] - fk[pass_f0], ymin=0, ymax=0.03, color=color_set[j],
                                        label=str(j+1) + " harmonic" + str(np.sqrt(energy_set[h, j])))
                    plt.legend()

                    plt.show()

    print("data set: ", dataset.shape)


if __name__ == '__main__':
    dir = r"C:\Users\邢\Desktop\10 nguru recording"
    dir1 = r"C:\Users\邢\Desktop\2recording"
    # extract_All_feature(dir)
    new_feature(dir, False, method='median', write_path = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_16_median.csv")
    # test_e(dir)



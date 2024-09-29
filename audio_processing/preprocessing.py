#librosa.effects.trim 去首尾静音
#librosa.effects.split
import math

import librosa
import librosa.display
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import os
import soundfile

def get_peak(array, threshold):
    for i in range(array.shape[0]):
        if(i >=threshold and i <= array.shape[0] - threshold -1):
            if(array[i] >= np.max(array[(i-threshold):(i+threshold + 1)])):
                peak = i
                break
    return peak

def pro_attack(rms, maxrms, thre_num):
    len = rms.shape[0]
    # if the attack part is correct
    for j in range(0, len, 1):
        if (rms[j] > maxrms * 0.2):
            attack_start = j
            break
    for j in range(0, len, 1):
        if (rms[j] >= maxrms * 0.9):
            attack_end = j
            break
    # rms = rms[attack_start:]
    # if ((attack_end - attack_start)/len < 0.1):
    #     return attack_start, attack_end
    #
    # attack_end = 1

    #Adaptive method for incorrect
    # the parameter for threshold
    M = 3
    # get w set
    w_set = np.zeros(thre_num)
    start_set = np.zeros(thre_num)
    thre_set = np.linspace(0,maxrms, thre_num)

    thre_num_20 = int(thre_num/5)
    start = attack_start
    for i in range(thre_num_20,thre_num,1):
        for j in range(attack_start, rms.shape[0], 1):
            if(rms[j] >= thre_set[i]):
                end = j
                w_set[i] = end - start
                start_set[i] = start
                start = end
                break
    # get average w, and then get attack part
    avg_w = np.mean(w_set[thre_num_20:])
    # print(w_set)
    # print(start_set)
    # print(avg_w)
    # if_not = False
    for i in range(thre_num_20,thre_num,1):
        # if(if_not):
        #     if (w_set[i] < M * avg_w):
        #         attack_start = start_set[i]
        #         if_not = True
        # else:
        if (w_set[i] >= M * avg_w):
            # print("get")
            attack_end = start_set[i]
            break
    return attack_start, int(attack_end)

def preprecessing12(path):
    n_fft = 512
    top_db = 30
    hop_length = int(n_fft / 4)
    duration = librosa.get_duration(filename=path)
    print(duration)
    y, sr = librosa.load(path, sr=None)
    stft = librosa.stft(y, window=np.ones, center=False)
    # print("stft: ", stft.shape)
    intervals = librosa.effects.split(y, top_db = top_db)
    new_interval = []
    win_interval = []
    attack = []
    five_part = []

    for ele in intervals:
        if (ele[1] - ele[0] > 1 / 2 * sr):
            ele[0] = ele[0] - 5000
            ys = y[ele[0]:ele[1]]
            S = librosa.magphase(librosa.stft(ys, n_fft = n_fft, hop_length = hop_length, window=np.ones, center=False))[0]
            # print(S.shape)
            times = librosa.times_like(S, n_fft = n_fft, hop_length = hop_length)
            rms = librosa.feature.rms(S = S, frame_length = n_fft, hop_length = hop_length)
            rms = np.squeeze(rms)
            # print("rms shape:", rms.shape)
            avgrms = np.mean(rms)
            maxrms = np.max(rms)
            interval_len = ele[1]-ele[0]
            weight = int(1 / 10 * interval_len)
            len = rms.shape[0]
            ranges = range(0, rms.shape[0])
            threshold = 0.2

            attack_start, attack_end = pro_attack(rms,maxrms, 200)
            # print("attack: ", attack_start, " to ", attack_end)
            for j in range(len - 1, -1, -1):
                if (rms[j] > maxrms * threshold):
                    max_in = j
                    break


            ele[0] = ele[0] + attack_start * hop_length
            ele[1] = ele[0] + max_in * hop_length
            new_interval.append(ele)
            if(ele[0] % hop_length >= 1/2 * hop_length):
                win_left = int(ele[0]/hop_length) + 1
            else:
                win_left = int(ele[0]/hop_length)
            if(ele[1] % hop_length >= 1/2 * hop_length):
                win_right = int(ele[1]/hop_length) + 1
            else:
                win_right = int(ele[1] / hop_length)

            fig, ax = plt.subplots()
            ax.plot(times, rms)
            ax.plot([times[attack_start], times[attack_start]], [0, maxrms], label='start', color='r')
            ax.plot([times[attack_end], times[attack_end]], [0, maxrms], label='end', color='g')
            ax.legend(loc='upper right')
            ax.set(title=str(ele[0]))
            plt.show()

            lenth = win_right - win_left
            five_part.append([win_left, win_left + round(1 * lenth/5), win_left + round(2 * lenth/5), win_left + round(3 * lenth/5), win_left + round(4 * lenth/5), win_left + lenth])
            win_interval.append([win_left,win_right])
            win_attact_end = win_left + attack_end
            attack.append(win_attact_end)


            # n_fft = 2048
            # fig, ax = plt.subplots()
            # stft = np.abs(librosa.stft(ys, n_fft=n_fft, hop_length=hop_length))
            # D = librosa.amplitude_to_db(stft)
            # img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=sr, n_fft=n_fft,
            #                                hop_length=n_fft / 4)
            # fig.colorbar(img, ax=ax, format="%+2.f dB")
            # plt.show()
            # print(attack_start)
            # print(attack_end)
            # print((attack_end - attack_start) / lenth)
    five_part = np.array(five_part)
    attack = np.array(attack)
    final_interval = np.array(new_interval)
    win_interval = np.array(win_interval)

    five_part = np.round(five_part / 4).astype(int)
    attack = np.round(attack / 4).astype(int)
    win_interval = np.round(win_interval / 4).astype(int)

    return final_interval, win_interval, attack, five_part

# to draw a picture describing weakest effort method in attack finding
def precess_onenote(path, n_fft = 1024, hop_length = 256):
    top_db = 40
    y, sr = librosa.load(path, sr=None)
    stft = librosa.stft(y, window=np.ones, center=False)
    # print("stft: ", stft.shape)
    intervals = librosa.effects.split(y, top_db=top_db)
    first_note = intervals[0]
    print("first note: ", first_note)
    ys = y[first_note[0] - 3000: first_note[1]]

    S = librosa.magphase(librosa.stft(ys, n_fft=n_fft, hop_length=hop_length, window=np.ones, center=False))[0]
    # print(S.shape)
    times = librosa.times_like(S, n_fft=n_fft, hop_length=hop_length)
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop_length)
    rms = np.squeeze(rms)
    maxrms = np.max(rms)
    len = rms.shape[0]
    threshold = np.arange(0.1, 1.1, 0.1)
    print(threshold)
    fig, ax = plt.subplots()
    y_list = []
    # 绘制折线图
    ax.plot(times, rms)
    ax.set_xlim([0, max(times)])
    ax.set_ylim([0, 1.02 * maxrms])

    attack_start, attack_end = pro_attack(rms, maxrms, 10)
    ax.axvline(x=times[attack_start], ymin=0, ymax=20, color= 'blue', linestyle= '--', label='start')
    ax.axvline(x=times[attack_end], ymin=0, ymax=20, color='red', linestyle='--', label='end')
    # draw help line
    for i in range(10):
        point_y = threshold[i] * maxrms  # need y pos
        y_list.append(point_y)
        print("y: ", point_y)
        # 查找第一次达到目标值的位置
        mask = np.greater_equal(rms, point_y)  # 比较y与目标值，生成布尔数组
        idx = np.argmax(mask)
        point_x = times[idx]
        print("x: ", point_x)
        ax.plot([point_x, point_x], [0, point_y], color='black')  # 绘制竖直的线
        ax.plot([0, point_x], [point_y, point_y], color='black')
    y_list = np.array(y_list)
    y_labels = ['$\\theta_{%s} = 0.%s$' % (i,i) for i in range(1,11,1)]
    y_labels = ['$\\theta_{1} = 0.1$', '$\\theta_{2} = 0.2$', '$\\theta_{3} = 0.3$', '$\\theta_{4} = 0.4$', '$\\theta_{5} = 0.5$', '$\\theta_{6} = 0.6$', '$\\theta_{7} = 0.7$', '$\\theta_{8} = 0.8$', '$\\theta_{9} = 0.9$', '$\\theta_{10} = 1.0$']
    ax.set_yticks([i for i in np.arange(0.1 * maxrms, 1.1 * maxrms, 1/10 * maxrms)])
    ax.set_yticklabels(y_labels)
    print(attack_start)
    print(rms[attack_start])
    ax.legend()

    plt.show()

def preprecessing1(y, sr, n_fft, top_db, output_hop_length):
    hop_length = int(n_fft / 4)
    output_factor = output_hop_length / hop_length
    stft = librosa.stft(y, window=np.ones, center=False)
    # print("stft: ", stft.shape)
    intervals = librosa.effects.split(y, top_db = top_db)
    new_interval = []
    win_interval = []
    attack = []
    five_part = []

    for ele in intervals:
        if (ele[1] - ele[0] > 1 / 2 * sr):
            ele[0] = ele[0] - 5000
            ys = y[ele[0]:ele[1]]
            S = librosa.magphase(librosa.stft(ys, n_fft = n_fft, hop_length = hop_length, window=np.ones, center=False))[0]
            # print(S.shape)
            times = librosa.times_like(S, n_fft = n_fft, hop_length = hop_length)
            rms = librosa.feature.rms(S = S, frame_length = n_fft, hop_length = hop_length)
            rms = np.squeeze(rms)
            # print("rms shape:", rms.shape)
            avgrms = np.mean(rms)
            maxrms = np.max(rms)
            interval_len = ele[1]-ele[0]
            weight = int(1 / 10 * interval_len)
            len = rms.shape[0]
            ranges = range(0, rms.shape[0])
            #the threshold of attack part if 0.2 of the max rms
            threshold = 0.2

            attack_start, attack_end = pro_attack(rms,maxrms, 200)
            # print("attack: ", attack_start, " to ", attack_end)
            for j in range(len - 1, -1, -1):
                if (rms[j] > maxrms * threshold):
                    max_in = j
                    break


            ele[0] = ele[0] + attack_start * hop_length
            ele[1] = ele[0] + max_in * hop_length
            new_interval.append(ele)
            if(ele[0] % hop_length >= 1/2 * hop_length):
                win_left = int(ele[0]/hop_length) + 1
            else:
                win_left = int(ele[0]/hop_length)
            if(ele[1] % hop_length >= 1/2 * hop_length):
                win_right = int(ele[1]/hop_length) + 1
            else:
                win_right = int(ele[1] / hop_length)
            #
            # fig, ax = plt.subplots()
            # ax.plot(times, rms)
            # ax.vlines(x = times[attack_start], ymin=0, ymax=maxrms, label='start ' + str(attack_start), color='r')
            # ax.vlines(x = times[attack_end], ymin=0, ymax=maxrms, label='attack_end' + str(attack_end), color='g')
            # ax.vlines(x=times[max_in], ymin=0, ymax=maxrms, label='end' + str(attack_end), color='y')
            # ax.legend(loc='upper right')
            # ax.set(title=str(ele[0]))
            # plt.show()

            lenth = win_right - win_left
            five_part.append([win_left, win_left + round(1 * lenth/5), win_left + round(2 * lenth/5), win_left + round(3 * lenth/5), win_left + round(4 * lenth/5), win_left + lenth])
            win_interval.append([win_left,win_right])
            win_attact_end = win_left + attack_end
            attack.append(win_attact_end)


            # n_fft = 2048
            # fig, ax = plt.subplots()
            # stft = np.abs(librosa.stft(ys, n_fft=n_fft, hop_length=hop_length))
            # D = librosa.amplitude_to_db(stft)
            # img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=sr, n_fft=n_fft,
            #                                hop_length=n_fft / 4)
            # fig.colorbar(img, ax=ax, format="%+2.f dB")
            # plt.show()
            # print(attack_start)
            # print(attack_end)
            # print((attack_end - attack_start) / lenth)

    five_part = np.array(five_part)
    attack = np.array(attack)
    final_interval = np.array(new_interval)
    win_interval = np.array(win_interval)

    five_part = np.round(five_part / output_factor).astype(int)
    attack = np.round(attack / output_factor).astype(int)
    win_interval = np.round(win_interval / output_factor).astype(int)
    print(win_interval.shape)
    if(win_interval.shape[0] < 60):
        top_db = 30
        return preprecessing1(y,sr,n_fft, top_db, output_hop_length)
    return final_interval, win_interval, attack, five_part


def pre_win(y, sr, hop_length):
    return

def prepro(y,sr,hop_length):
    n_fft = 2048
    top_db = 30
    hop_length = int(n_fft / 4)
    stft = librosa.stft(y, window=np.ones, center=False)
    print("stft: ", stft.shape)
    intervals = librosa.effects.split(y, top_db=top_db)
    new_interval = []
    win_interval = []

    for ele in intervals:
        if (ele[1] - ele[0] > 1 / 2 * sr):
            ele[0] = ele[0] - 5000
            ys = y[ele[0]:ele[1]]
            S = librosa.magphase(librosa.stft(ys, window=np.ones, center=False))[0]
            print(S.shape)
            times = librosa.times_like(S)
            rms = librosa.feature.rms(S=S)
            rms = np.squeeze(rms)
            avgrms = np.mean(rms)
            maxrms = np.max(rms)
            interval_len = ele[1] - ele[0]
            weight = int(1 / 10 * interval_len)
            len = rms.shape[0]
            ranges = range(0, rms.shape[0])
            threshold = 0.95
            for j in range(0, len, 1):
                if (rms[j] > avgrms * threshold):
                    min_in = j
                    break
            for j in range(len - 1, -1, -1):
                if (rms[j] > avgrms * threshold):
                    max_in = j
                    break
            ele[1] = ele[0] + max_in * hop_length
            ele[0] = ele[0] + min_in * hop_length
            new_interval.append(ele)
            if (ele[0] % hop_length >= 1 / 2 * hop_length):
                win_left = int(ele[0] / hop_length) + 1
            else:
                win_left = int(ele[0] / hop_length)
            if (ele[1] % hop_length >= 1 / 2 * hop_length):
                win_right = int(ele[1] / hop_length) + 1
            else:
                win_right = int(ele[1] / hop_length)
            win_interval.append([win_left, win_right])

            fig, ax = plt.subplots()
            ax.plot(times, rms)
            plt.show()
            # n_fft = 2048
            # fig, ax = plt.subplots()
            # stft = np.abs(librosa.stft(ys, n_fft=n_fft, hop_length=hop_length))
            # D = librosa.amplitude_to_db(stft)
            # img = librosa.display.specshow(D, x_axis='time', y_axis='log', ax=ax, sr=sr, n_fft=n_fft,
            #                                hop_length=n_fft / 4)
            # fig.colorbar(img, ax=ax, format="%+2.f dB")
            # plt.show()

    final_interval = np.array(new_interval)
    win_interval = np.array(win_interval)


# cut one tone to five
def preprecessing(y,sr,hop_length):
    top_db = 30
    intervals = librosa.effects.split(y, top_db = top_db)
    new_interval = []
    win_interval = []
    for ele in intervals:
        if (ele[1] - ele[0] > 1 / 2 * sr):
            ys = y[ele[0]:ele[1]]
            S = librosa.magphase(librosa.stft(ys, window=np.ones, center=False))[0]
            rms = librosa.feature.rms(S = S)
            rms = np.squeeze(rms)
            avgrms = np.mean(rms)
            interval_len = ele[1]-ele[0]
            weight = int(1 / 10 * interval_len)
            len = rms.shape[0]
            ranges = range(0, rms.shape[0])
            threshold = 0.95
            for j in range(0, len, 1):
                if(rms[j] > avgrms * threshold):
                    min_in = j
                    break
            for j in range(len -1, -1, -1):
                if(rms[j] > avgrms * threshold):
                    max_in = j
                    break
            ele[1] = ele[0] + max_in * hop_length
            ele[0] = ele[0] + min_in * hop_length
            new_interval.append(ele)
            if(ele[0] % hop_length >= 1/2 * hop_length):
                win_left = int(ele[0]/hop_length) + 1
            else:
                win_left = int(ele[0]/hop_length)
            if(ele[1] % hop_length >= 1/2 * hop_length):
                win_right = int(ele[1]/hop_length) + 1
            else:
                win_right = int(ele[1] / hop_length)
            win_interval.append([win_left,win_right])

    final_interval = np.array(new_interval)
    win_interval = np.array(win_interval)
    return final_interval, win_interval

def process_audio(dir):
    # note name list
    note_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    note_7_8_list = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'J', 'K', 'L', 'M']
    note_10_List = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'N', 'O', 'P']
    dataset = np.array([])
    instrument_num = 1
    path_list = []
    for i in os.listdir(dir):
        file = os.path.splitext(i)
        if (file[1] == ".wav" or file[1] == ".mp3"):
            path_list.append(i)
    path_list.sort(key=lambda x: int(x[6:][:-4]))

    print(path_list)

    for i in path_list:
        file = os.path.splitext(i)
        # if (file[1] != ".wav" and file[1] != ".mp3"):
        #     continue;
        print(file)
        y, sr = librosa.load(os.path.join(dir, i), sr=None)
        length = len(y)

        n_fft = frame_length = 4096
        hop_length = int(n_fft / 4)
        final_interval, win_interval, attack, five_part = preprecessing1(y, sr, n_fft=1024, top_db=40,
                                                                                       output_hop_length=hop_length)
        # note_num: which num(according to note_List) this note belong to, iter: the position of this tone in this audio
        ab_path = r"C:\Users\xrw\Desktop\note_audio"
        for j,interval in enumerate(final_interval):
            y_note = y[interval[0]:interval[1]]
            note_iter = j % 5 + 1
            note_num = math.floor(j / 5)
            # instrument 2 has 81 notes
            if(note_num == 16):
                note_num = 15
                note_iter = 6
            if(file[0] == "nguru-7" or file[0] == "nguru-8"):
                note = note_7_8_list[note_num]
            elif (file[0] == "nguru-10"):
                note = note_10_List[note_num]
            else:
                note = note_list[note_num]
            note_info = file[0] + '-' + note + '-' + str(note_iter) + '.wav'
            note_path = os.path.join(ab_path, note_info)
            soundfile.write(note_path, y_note, sr)





# # 接下来最好分出音频来听一下，然后ok
if __name__ == '__main__':
    # final_interval, win_interval, attack, five_part = preprecessing1(path=r"C:\Users\邢\Desktop\10 nguru recording\nguru-8.wav")
    # print("final_interval: ", final_interval.shape)
    # print(final_interval)
    # print("win_interval: ", win_interval.shape)
    # print("attack: ", attack)
    dir = r"C:\Users\xrw\Desktop\10 nguru recording"
    # process_audio(dir)
    audio_path = r"C:\Users\xrw\Desktop\10 nguru recording\nguru-1.wav"
    path = r"C:\Users\xrw\Desktop\research result\final_result\final_result\5cluster\clustered_audio\cluster_0\nguru-1-J-1.wav"
    precess_onenote(audio_path)
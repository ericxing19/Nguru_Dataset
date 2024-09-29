import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def print_set(set):
    for j in range(0, set.shape[0], 100):
        print(j, " ", set[j])

def position(y):
    for i in range(y.shape[0]):
        y

# high time frequency, to get the voiced segment
def piptrack(path):
    n_fft = 2048
    hop_length = int(n_fft/4)
    duration = librosa.get_duration(filename = path)
    y, sr = librosa.load(path, sr=None)
    times = np.arange(0, duration, hop_length/sr)
    print(times.shape)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft = n_fft, hop_length = hop_length)
    print(pitches.shape)
    print(magnitudes.shape)
    time_series = np.zeros(times.shape)
    for i in range(0,time_series.shape[0],1):
        if any(pitches[:,i]):
            time_series[i] = 1
    j = 0
    m = 0
    interval = []
    for i in range(time_series.shape[0]):
        if (j == 0):
            start = i
            end = i
        j += 1
        end += 1
        if(j == 100):
            m += 1
        if(time_series[i] == 0):
            j = 0
            if(end - start >= 240):
                interval.append([start,end])

    interval = np.array(interval)
    print(interval.shape)
    print(interval)
    print(m)
    print(time_series)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(pitches, x_axis='time', y_axis='log', ax=ax, sr=sr, n_fft=n_fft,
                                   hop_length=n_fft / 4)
    ax.set(title='pitch')
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    fig2, ax1 = plt.subplots()
    img = librosa.display.specshow(magnitudes, x_axis='time', y_axis='log', ax=ax1, sr=sr, n_fft=n_fft,
                                   hop_length=n_fft / 4)
    ax1.set(title='magnitude')
    fig2.colorbar(img, ax=ax1, format="%+2.f dB")
    plt.show()
    # np.savetxt(path + "2048pitch.csv", pitches, delimiter=',')
    # np.savetxt(path + "2048mag.csv", magnitudes, delimiter=',')

if __name__ == '__main__':
    piptrack(path= r"C:\Users\邢\Desktop\10 nguru recording\nguru-2.wav")
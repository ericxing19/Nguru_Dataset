import numpy
import pandas as pd
import numpy as np
def read_data(path):
    data = pd.read_csv(path, header=None)
    data = np.array(data)
    data = data.reshape(data.shape[0], )
    print(data.shape)
    data = numpy.array(data)
    multi_f0 = numpy.zeros((data.shape[0],4))
    for i, str in enumerate(data):
        print(i)
        list = str.split("\t")
        for j in range(len(list)):
            multi_f0[i][j] = list[j]
    print(multi_f0.T)
    time_sr = 1024/48000
    print(multi_f0.shape)
    print(multi_f0[-1][0]/time_sr)

if __name__ == '__main__':
    read_data(r"C:\Users\邢\Desktop\second-b-2.csv")

#2293
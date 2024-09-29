import os.path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def p(path):
    data = pd.read_csv(path, sep=',')
    # data_without_NaN = data.dropna(axis=0)
    # print(data_without_NaN)
    my_imputer = SimpleImputer()
    data_imputed = my_imputer.fit_transform(data)
    print(type(data_imputed))
    # array转换成df
    df_data_imputed = pd.DataFrame(data_imputed, columns=data.columns)
    print(df_data_imputed)
    df_data_imputed.to_csv(r"C:\Users\邢\Desktop\10 nguru recording\2022_11_17_p.csv")


def p2(path):
    data = pd.read_csv(path, sep=',')
    co = data.columns
    d = data.values
    print(d.shape)
    for i in range(55,60):
        for j in range(d.shape[0]):
            if (np.isnan(d[j, i])):
                d[j,i] = 0
            if(np.isnan(d[j, 29])):
                d[j,29] = 0
    my_imputer = SimpleImputer()
    d_imputed = my_imputer.fit_transform(d)
    nd = pd.DataFrame(d_imputed, columns= co)
    print(nd)
    file = os.path.splitext(path)
    output_path = file[0] + '_processed' + file[1]
    print(output_path)
    nd.to_csv(output_path)


def process_note(path):
    data = pd.read_csv(path, sep=',')
    co = data.columns
    d = data.as_matrix()

def MR_scale(path):
    data = pd.read_csv(path, sep=',')
    co = data.columns
    X = data.values
    D = np.zeros(X.shape)
    for j in range(X.shape[1]-1):
        mean = np.mean(X[:, j])
        for i in range(X.shape[0]):
            D[i][j] = X[i][j] / mean
    new = pd.DataFrame(D, columns=co)
    new.to_csv(r"C:\Users\邢\Desktop\10 nguru recording\2022_9_21_processed_MRnormalized.csv")

def MVR_scale(path):
    data = pd.read_csv(path, sep=',')
    co = data.columns
    X = data.values
    D = np.zeros(X.shape)
    for j in range(X.shape[1]-1):
        mean = np.mean(X[:, j])
        std = np.std(X[:,j])
        for i in range(X.shape[0]):
            D[i][j] = (X[i][j] - mean)/ std
    new = pd.DataFrame(D, columns=co)
    new.to_csv(r"C:\Users\邢\Desktop\10 nguru recording\2022_9_21_processed_MVRnormalized.csv")




if __name__ == '__main__':
    path = r'C:\Users\邢\Desktop\10 nguru recording\2022_9_21.csv'
    mean_path = r"C:\Users\邢\Desktop\10 nguru recording\2022_11_17.csv"
    median_48000_path = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_8_median.csv"
    median_22050_path = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_15_median(22050hz).csv"
    median_4096_48000_path = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft).csv"
    # p(path)
    p2(median_4096_48000_path)
    # MR_scale(new_path)
    # MVR_scale(new_path)
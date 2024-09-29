import numpy as np

def Log_scale(X, if_instrument = 0, if_label = 0):
    D = X.copy()
    cut_index = 0
    if(if_instrument):
        cut_index += 1
    if(if_label):
        cut_index += 1
    for j in range(X.shape[1] - cut_index):
        for i in range(X.shape[0]):
            D[i][j] = np.log2(X[i][j]+1e-5)
    return D

# if_instrument = 0 means there's no instrument list, if_label as well
def MVR_scale(X, if_instrument = 0, if_label = 0):
    D = X.copy()
    # default to be 0, when there's no instrument and label
    cut_index = 0
    if(if_instrument):
        cut_index += 1
    if(if_label):
        cut_index += 1
    for j in range(X.shape[1] - cut_index):
        mean = np.mean(X[:, j])
        std = np.std(X[:,j])
        for i in range(X.shape[0]):
            D[i][j] = (X[i][j] - mean)/ std
    return D

def MR_scale(X, if_instrument = 0, if_label = 0):
    D = X.copy()
    cut_index = 0
    if(if_instrument):
        cut_index += 1
    if(if_label):
        cut_index += 1
    for j in range(X.shape[1] - cut_index):
        mean = np.mean(X[:, j])
        for i in range(X.shape[0]):
            D[i][j] = X[i][j] / mean
    return D

def RR_scale(X, if_instrument = 0, if_label = 0):
    D = X.copy()
    cut_index = 0
    if(if_instrument):
        cut_index += 1
    if(if_label):
        cut_index += 1
    for j in range(X.shape[1] - cut_index):
        max = np.max(X[:, j])
        min = np.min(X[:, j])
        for i in range(X.shape[0]):
            D[i][j] = (X[i][j] - min) / (max - min)
    return D

if __name__ == '__main__':
    pass
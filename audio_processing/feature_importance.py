from sklearn.datasets import load_diabetes
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import svm

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import Standardization

csv_path = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed34.csv"

dataset = pd.read_csv(csv_path)
data_columns = dataset.columns
dataset = dataset.values
nt_feature = 60

print("data: ", dataset.shape)
# 1.four or one method  2. whether draw the picture 3. whether draw the average picture 4. whether write cluster number 5.whether show instrument list
# One method
if_plot = [False, False, False, False, False]
# four method
# if_plot = [True, True, False, False, True]
# one method, average picture
# if_plot = [False, False, True, False, True]
if (dataset.shape[1] == nt_feature + 1):
    data = dataset[:, :-1]
    instrument_list = dataset[:, -1]
    data_columns = data_columns[:,-1]
elif (dataset.shape[1] == nt_feature + 2):
    data = dataset[:, :-2]
    instrument_list = dataset[:, -2]
    data_columns = data_columns[:-2]
    labels = dataset[:, -1]
instrument_list = np.squeeze(instrument_list)
MR_data = Standardization.MR_scale(data)

except_absolute = np.delete(MR_data, np.arange(0, 8, 1), axis=1)

except_absolute_4and5th_harmonic = np.delete(except_absolute,
                                             (2, 3, 8, 9, 13, 14, 19, 20, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46),
                                             axis=1)
n_feature = 34
except_absolute_la = np.delete(data_columns, np.arange(0, 8, 1))
except_absolute_4and5th_harmonic_la = np.delete(except_absolute_la,
                                                (2, 3, 8, 9, 13, 14, 19, 20, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46))

X_train, X_val, y_train, y_val = train_test_split(except_absolute_4and5th_harmonic, labels, random_state=0)

print("labels: ", labels)
print("columns: ", except_absolute_4and5th_harmonic_la)

def permutation_fim(model_used):
    model = svm.SVC()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_val)
    print(y_pre)
    print(y_val)
    print(y_pre == y_val)
    print("accuracy: ", np.sum(y_pre == y_val)/len(y_pre))
    scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    r_multi = permutation_importance(model, X_val, y_val, n_repeats=30, random_state=0, scoring=scoring)
    for metric in r_multi:
        print(f"{metric}")
        r = r_multi[metric]
        print()
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"    {except_absolute_4and5th_harmonic_la[i]:<8}: "
                      f"{r.importances_mean[i]:.3f}"
                      f" +/- {r.importances_std[i]:.3f}")

if __name__ == '__main__':
    permutation_fim()


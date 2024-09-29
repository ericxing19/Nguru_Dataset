import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import Standardization


# Removing the outliers
def removeOutliers(data, col):
    Q3 = np.quantile(data[col], 0.75)
    Q1 = np.quantile(data[col], 0.25)
    IQR = Q3 - Q1
    if(IQR == 0 or IQR == np.nan):
        return data
    print("IQR value for column %s is: %s" % (col, IQR))

    lower_range = Q1 - 6 * IQR
    upper_range = Q3 + 6 * IQR
    outlier_free_list = [x for x in data[col] if (
            (x >= lower_range) & (x <= upper_range))]
    filtered_data = data.loc[data[col].isin(outlier_free_list)]
    print(filtered_data)
    return filtered_data

def remove_alloutliers(data, if_have_label):
    if(if_have_label):
        for i in data.columns:
            if i == data.columns[0]:
                print(i)
                filtered_data = removeOutliers(data, i)
            elif i == "instrument" or i == "cluster":
                pass
            else:
                filtered_data = removeOutliers(filtered_data, i)
    else:
        for i in data.columns:
            if i == data.columns[0]:
                filtered_data = removeOutliers(data, i)
            else:
                filtered_data = removeOutliers(filtered_data, i)
    return filtered_data


# dataset_co = odataset.columns
# data = odataset.values
# data = np.delete(data, (221,523,755), axis=0)
# dataset = pd.DataFrame(data, columns=dataset_co)
print("go")
# newd = dataset[['avg_zcr'， 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio sp/1', 'harmonic energy ratio1', 'harmonic energy ratio2','harmonic energy ratio3', 'harmonic energy ratio sp', 'cluster']]

# sns.pairplot(dataset, diag_kind='kde', hue = 'cluster')
# plt.title("34feature distribution")
# plt.savefig(r"C:\Users\xrw\Desktop\filtered_2023_2_16_median 34feature distribution")
def log_show_dis(dataset):
    data_col = dataset.columns
    col = data_col[:-2]
    print(col)
    for i in col:
        dataset[i] = np.log(dataset[i]+1e-5)
    sns.pairplot(dataset[['avg_zcr', 'att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'harmonic energy ratio1', 'harmonic energy ratio2','harmonic energy ratio3', 'harmonic energy ratio sp', 'cluster']], diag_kind='kde', hue = 'cluster')
    # sns.pairplot(dataset, diag_kind='kde', hue='cluster')
    plt.title("log feature distribution")
    plt.savefig(r"C:\Users\xrw\Desktop\log_feature distribution")

def MR_show_dis(dataset):
    data_col = dataset.columns
    col = data_col[:-2]
    print(col)
    X = dataset.values
    MR_X = Standardization.MR_scale(X, True, True)
    MR_dataset = pd.DataFrame(MR_X, columns = dataset.columns)
    print(MR_dataset)
    sns.pairplot(MR_dataset[['avg_zcr', 'att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'harmonic energy ratio1', 'harmonic energy ratio2','harmonic energy ratio3', 'harmonic energy ratio sp', 'cluster']], diag_kind='kde', hue='cluster')
    plt.title("MR feature distribution")
    plt.savefig(r"C:\Users\xrw\Desktop\MR_feature distribution")

def MVR_show_dis(dataset):
    data_col = dataset.columns
    col = data_col[:-2]
    print(col)
    X = dataset.values
    MVR_X = Standardization.MVR_scale(X, True, True)
    MVR_dataset = pd.DataFrame(MVR_X, columns = dataset.columns)
    print(MVR_dataset)
    sns.pairplot(MVR_dataset[['harmonic energy ratio1', 'harmonic energy ratio2','harmonic energy ratio3', 'harmonic energy ratio4', 'harmonic energy ratio5', 'harmonic energy ratio sp', 'att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'att harmonic ratio4', 'att harmonic ratio5', 'att harmonic ratio sp', '1st harmonic1', '2nd harmonic1', '3rd harmonic1', '4th harmonic1', '5th harmonic1', '1st harmonicsp', '2nd harmonicsp', '3rd harmonicsp', '4th harmonicsp', '5th harmonicsp', 'cluster']], diag_kind='kde', hue='cluster')
    plt.title("MVR 34feature distribution")
    plt.savefig(r"C:\Users\xrw\Desktop\MVR_2023_2_16_median 34feature distribution")

# d = dataset[['avg_zcr', 'total_energy', 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio sp/1', 'harmonic energy ratio1', 'harmonic energy ratio2','harmonic energy ratio3', 'harmonic energy ratio sp', 'instrument']]
# # a = dataset[['avg_zcr', 'total_energy', 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio4/1', 'pitch ratio5/1', 'pitch ratio sp/1', 'harmonic energy ratio1', 'harmonic energy ratio2','harmonic energy ratio3','harmonic energy ratio4','harmonic energy ratio5', 'harmonic energy ratio sp']]
# b = dataset[['avg_zcr', 'total_energy', 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio4/1', 'pitch ratio5/1', 'pitch ratio sp/1', 'attack pitch ratio2/1', 'attack pitch ratio3/1', 'attack pitch ratio4/1', 'attack pitch ratio5/1', 'attack pitch ratio sp/1', 'att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'att harmonic ratio4', 'att harmonic ratio5', 'att harmonic ratio sp', 'instrument']]
# # c = b = dataset[['avg_zcr', 'total_energy', 'pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio4/1', 'pitch ratio5/1', 'attack pitch ratio2/1', 'attack pitch ratio3/1', 'attack pitch ratio4/1', 'attack pitch ratio5/1', 'att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'att harmonic ratio4', 'att harmonic ratio5']]
# c = dataset[['harmonic energy ratio1', 'harmonic energy ratio2','harmonic energy ratio3', 'harmonic energy ratio4', 'harmonic energy ratio5', 'harmonic energy ratio sp', 'att harmonic ratio1', 'att harmonic ratio2', 'att harmonic ratio3', 'att harmonic ratio4', 'att harmonic ratio5', 'att harmonic ratio sp', '1st harmonic1', '2nd harmonic1', '3rd harmonic1', '4th harmonic1', '5th harmonic1', '1st harmonic2', '2nd harmonic2', '3rd harmonic2', '4th harmonic2', '5th harmonic2', '1st harmonicsp', '2nd harmonicsp', '3rd harmonicsp', '4th harmonicsp', '5th harmonicsp', 'instrument']]
# print("c: ", c)
# data_filtered = remove_alloutliers(dataset, True)
# d_filtered = remove_alloutliers(d, True)
# b_filtered = remove_alloutliers(b, True)
# c_filtered = remove_alloutliers(c, True)
# print("data_filtered: ", data_filtered)
# print("d_filtered: ", d_filtered)
# print("b_filtered: ", b_filtered)
# print("c_filtered: ", c_filtered)
# sns.pairplot(d, diag_kind='kde', hue = 'instrument')
# plt.title("ori 10 feature distribution")
# plt.savefig(r"C:\Users\邢\Desktop\filtered_2023_2_16_median_ori 10 feature distribution")
# sns.pairplot(d_filtered, diag_kind='kde', hue = 'instrument')
# plt.title("filtered 10 feature distribution")
# plt.savefig(r"C:\Users\邢\Desktop\filtered_2023_2_16_median_filtered 10 feature distribution")
# sns.pairplot(b_filtered, diag_kind='kde', hue = 'instrument')
# plt.title("filtered 20 feature distribution")
# plt.savefig(r"C:\Users\邢\Desktop\filtered_2023_2_16_median_filtered 20 feature distribution")





# sns.pairplot(c_filtered, diag_kind='kde', hue = 'instrument')
# plt.title("harmonic distribution")
# plt.savefig(r"C:\Users\邢\Desktop\filtered_2023_2_16_median harmonic feature distribution")


# # pca c
# c_data = c_filtered.values[:, :-1]
# pca = decomposition.PCA(n_components=10)
# pca.fit(c_data)
# trans_data = pca.transform(c_data)
# print("pca: ", pca.explained_variance_ratio_)
# pca_Data2 = pandas.DataFrame(trans_data)
# print(pca_Data2)

# # MVR
# MVR_data = Standardization.MVR_scale(data_filtered)
# MVR_data = pandas.DataFrame(MVR_data)
# sns.pairplot(MVR_data,diag_kind='kde')
# plt.savefig(r"C:\Users\邢\Desktop\filtered_2023_2_8_median MVR10 feature distribution")
#
#
# sns.pairplot(pca_Data2,diag_kind='kde')
# plt.savefig(r"C:\Users\邢\Desktop\filtered_2023_2_8_median harmonic pca10 feature distribution")
if __name__ == '__main__':
    csv_path = r"C:\Users\邢\Desktop\10 nguru recording\2022_9_21_processed.csv"
    csv_path2 = r"C:\Users\邢\Desktop\10 nguru recording\2022_11_17_processed.csv"
    csv_path3 = r"C:\Users\邢\Desktop\10 nguru recording\2022_11_18_cqt.csv"
    csv_path4 = r"C:\Users\邢\Desktop\10 nguru recording\2022_11_18_log_cqt.csv"
    csv_path5 = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_6_median_processed.csv"
    csv_path6 = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_8_median_processed.csv"
    csv_path7 = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_15_median(22050hz)_processed.csv"
    csv_path8 = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed.csv"
    csv_path9 = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed34.csv"
    csv_path10 = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed34log_gmm_328.csv"
    csv_path11 = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed34.csv"
    csv_path12 = r"C:\Users\xrw\Desktop\research result\final_result\5cluster\4_14_10_5cluster.csv"
    dataset = pd.read_csv(csv_path12)
    datasetMR = pd.read_csv(csv_path11)
    log_show_dis(dataset)
    MR_show_dis(datasetMR)
    plt.show()

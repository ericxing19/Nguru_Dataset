import numpy as np
import pandas as pd

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

# remove using dataframes
def remove_alloutliers_d(data, if_have_label):
    if(if_have_label):
        for i in data.columns:
            if i == data.columns[0]:
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


# remove using matrix
def remove_alloutliers_m(data_value, if_have_label, data_columns):
    data = pd.DataFrame(data_value, columns=data_columns)
    if(if_have_label):
        for i in data_columns:
            if i == data_columns[0]:
                filtered_data = removeOutliers(data, i)
            elif i == "instrument" or i == "cluster":
                pass
            else:
                filtered_data = removeOutliers(filtered_data, i)
    else:
        for i in data_columns:
            if i == data_columns[0]:
                filtered_data = removeOutliers(data, i)
            else:
                filtered_data = removeOutliers(filtered_data, i)
    return filtered_data.values
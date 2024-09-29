import os.path

import numpy as np
import pandas as pd
import sklearn
import itertools
import Standardization
import csv
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
from sklearn import mixture
from remove_outliers import remove_alloutliers_m

n_cluster = 11
# hc = sklearn.cluster.AgglomerativeClustering(linkage='ward', n_clusters = n_cluster).fit(data)
# true: hie, false：GMM
hie_or_gmm = False

if not(hie_or_gmm):
  # csv_path = r"C:\Users\邢\Desktop\10 nguru recording\totaldataset_18_t.csv"
  # csv_path = r"C:\Users\邢\Desktop\10 nguru recording\5note_dataset_18_t.csv"
  # csv_path = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_8_median_processed.csv"
  csv_path = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed.csv"
  front_path, end_path = os.path.splitext(csv_path)
  # csv_path = r"C:\Users\邢\Desktop\10 nguru recording\finalset_t.csv"
else:
  # csv_path = r"C:\Users\邢\Desktop\posterior.csv"
  # csv_path = r"C:\Users\邢\Desktop\10 nguru recording\2023_2_8_median_processed.csv"
  csv_path = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed.csv"

def split_train_test(data):
    note_number = 151
    test_index = []
    for i in range(note_number):
        test_index.append(np.random.randint(5 * i, 5 * (i+1)))
    test_data = data[test_index]
    train_data = np.delete(data,test_index,axis=0)
    return np.array(train_data), np.array(test_data)

def write_cluster(path, cluster_labels, n_feature):
    i = 0
    n_feature = 60
    data = pd.read_csv(csv_path)
    rows = data.values
    print(data.columns)
    print(rows.shape)
    print(n_feature)

    f = open(path, 'w+', newline = '')
    writer = csv.writer(f)
    if (rows.shape[1] == n_feature + 1):
        writer.writerow(np.append(data.columns, 'cluster'))
    elif (rows.shape[1] == n_feature + 2):
        rows = rows[:, :-1]
        writer.writerow(data.columns)
    for row in rows:
        new_row = np.append(row,cluster_labels[i])
        i += 1
        writer.writerow(new_row)
    f.close()


# def gmm(X, test_data, if_plot, n_feature):
#     train_num = X.shape[0]
#     test_num = test_data.shape[0]
#     lowest_bic = np.infty
#     bic = []
#     n_components_range = range(2, n_cluster)
#     if(if_plot[0]):
#       # cv_types = ["spherical", "tied", "diag", "full"]
#       cv_types = ["spherical", "diag"]
#     else:
#         # cv_types = ["full"]
#         cv_types = ["diag"]
#
#     for cv_type in cv_types:
#         silhouette_score_set = []
#         model_likelihood_list = []
#         for n_components in n_components_range:
#             # Fit a Gaussian mixture with EM
#             gmm = mixture.GaussianMixture(
#                 n_components=n_components, covariance_type=cv_type, max_iter = 300, tol = 1e-4
#             )
#             gmm.fit(X)
#             bic.append(gmm.bic(test_data))
#             train_cluster_labels = gmm.predict(X)
#             test_cluster_labels = gmm.predict(test_data)
#             if bic[-1] < lowest_bic:
#                 lowest_bic = bic[-1]
#                 best_gmm = gmm
#                 best_component = n_components
#                 best_method = cv_type
#             model_likelihood = -gmm.score(test_data)
#             sample_likelihood = -gmm.score_samples(test_data)
#             model_likelihood_list.append(model_likelihood)
#         if (if_plot[4]):
#             cluster_iter = 0
#             train_cluster = np.zeros(best_component,)
#             for i in range(best_component):
#                 for j in range(train_num):
#                     if (train_cluster_labels[j] == cluster_iter):
#                         train_cluster[i] += 1
#                 cluster_iter += 1
#
#             cluster_iter = 0
#             test_cluster = np.zeros(best_component,)
#             for i in range(best_component):
#                 for j in range(test_num):
#                     if(test_cluster_labels[j] == cluster_iter):
#                         test_cluster[i] += 1
#                 cluster_iter += 1
#             print(train_cluster)
#             plt.subplot(2, 1, 1)
#             plt.xticks(range(0, best_component))
#             plt.bar(range(len(train_cluster)), train_cluster)
#             plt.title(str(n_feature) + " features " + cv_type + ' train set ' + " cluster result")
#
#
#             # print(ins_cluster)
#             plt.subplot(2, 1, 2)
#             plt.xticks(range(0, best_component))
#             plt.bar(range(len(test_cluster)), test_cluster)
#             plt.title(str(n_feature) + " features " + cv_type + 'test set' + " cluster result")
#             plt.show()
#
#
#     bic = np.array(bic)
#     # for i in range(0,n_cluster-2):
#     #   print(i+2, "clusters: ", bic[i])
#     color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
#     clf = best_gmm
#     best_cluster_num = best_component
#     bars = []
#
#
#     if(if_plot[1]):
#         plt.figure(figsize=(8, 6))
#         spl = plt.subplot(2, 1, 1)
#         if(if_plot[0]):
#             for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#                 xpos = np.array(n_components_range) + 0.2 * (i - 2)
#                 bars.append(
#                     plt.bar(
#                         xpos,
#                         bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
#                         width=0.2,
#                         color=color,
#                     )
#                 )
#         if not (if_plot[0]):
#             for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#                     xpos = np.array(n_components_range)
#                     bars.append(
#                         plt.bar(
#                             xpos,
#                             bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
#                             width=0.2,
#                             color=color,
#                         )
#                     )
#         plt.xticks(n_components_range)
#         plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
#         plt.title(str(n_feature) + " features BIC score per model")
#         xpos = (
#                 np.mod(bic.argmin(), len(n_components_range))
#                 + 0.65
#                 + 0.2 * np.floor(bic.argmin() / len(n_components_range))
#         )
#         plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
#         spl.set_xlabel(str(n_feature) + " features Number of components")
#         spl.legend([b[0] for b in bars], cv_types)
#
#         splt = plt.subplot(2, 1, 2)
#         print(model_likelihood_list)
#         plt.title("model likelihood")
#         plt.xticks(n_components_range)
#         plt.plot(n_components_range, model_likelihood_list)
#     plt.show()
#     # print("best cluster numbers = ", best_cluster_num, ", value = ", clf, ", best method = ", best_method, 'lowest_bic: ', lowest_bic)
#     density = best_gmm.predict_proba(X)
#     print(np.max(density, axis=1))
#     # print(best_gmm.weights_)
#     return bic, [best_component, best_gmm, lowest_bic]

def plot_cluster_number_dis(best_component, instrument_list, best_cluster_labels, n_feature, cv_type='full'):
    ins_cluster = []
    instrument_num = 10
    cluster_num = best_component
    instrument_l = []
    for i in range(cluster_num):
        instrument_l.append("cluster " + str(i))
    cluster_iter = 0
    for i in range(best_component):
        in_list = []
        for j in range(instrument_list.shape[0]):
            if (best_cluster_labels[j] == cluster_iter):
                in_list.append(instrument_list[j])
        cluster_iter += 1
        ins_cluster.append(in_list)
    ins_cluster = np.array(ins_cluster)
    print(ins_cluster.shape)
    plt.subplot(2, 1, 1)
    plt.xticks(range(1, instrument_num + 1))
    plt.legend(loc='upper right')
    plt.hist(ins_cluster, bins=10, edgecolor="r", alpha=0.5, label=instrument_l)
    plt.title(str(n_feature) + " features " + cv_type + " instrument cluster result")
    #     best_component_list.append("cluster" + str(i))
    # for i in range(len(instrument_l)):


    # print(ins_cluster)
    plt.subplot(2, 1, 2)
    plt.xticks(range(0, best_component))

    plt.hist(best_cluster_labels, bins=10)
    plt.title(str(n_feature) + " features " + cv_type + " cluster result")

    ins_clusters = np.zeros((10, best_component))
    instrument_iter = 1
    for j in range(0, instrument_num):
        for i in range(instrument_list.shape[0]):
            if (int(instrument_list[i]) == instrument_iter):
                cluster_num = best_cluster_labels[i]
                ins_clusters[instrument_iter - 1][cluster_num] += 1
        instrument_iter += 1

# cluster trained on train_set, test on test_set
def gmm_t(total_data, X, test_data, if_plot, instrument_list, n_feature):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(6, 7)
    if(if_plot[0]):
        cv_types = ["diag", "full"]
    else:
        # cv_types = ["full"]
        cv_types = ["diag"]

    for cv_type in cv_types:
        silhouette_score_set = []
        model_likelihood_list = []
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type, max_iter = 200, tol = 7e-4
            )
            gmm.fit(X)
            bic.append(gmm.bic(test_data))
            cluster_labels = gmm.predict(total_data)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_component = n_components
                best_method = cv_type
                best_cluster_labels = cluster_labels
            model_likelihood = gmm.score(test_data)
            sample_likelihood = gmm.score_samples(test_data)
            print("sample likelihood: ", sample_likelihood)
            density = gmm.predict_proba(test_data)
            print("density: ", density)
            model_likelihood_list.append(model_likelihood)
        if (if_plot[4]):
            plot_cluster_number_dis(best_component, instrument_list, best_cluster_labels, n_feature, cv_type)
            # print("best cluster labels: ", best_cluster_labels)

    bic = np.array(bic)
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    best_cluster_num = best_component
    bars = []


    if(if_plot[1]):
        plt.figure(figsize=(8, 6))
        spl = plt.subplot(2, 1, 1)
        if(if_plot[0]):
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                xpos = np.array(n_components_range) + 0.2 * (i - 2)
                bars.append(
                    plt.bar(
                        xpos,
                        bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
                        width=0.2,
                        color=color,
                    )
                )
        if not (if_plot[0]):
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                    xpos = np.array(n_components_range)
                    bars.append(
                        plt.bar(
                            xpos,
                            bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
                            width=0.2,
                            color=color,
                        )
                    )
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
        plt.title(str(n_feature) + " features BIC score per model")
        xpos = (
                np.mod(bic.argmin(), len(n_components_range))
                + 0.65
                + 0.2 * np.floor(bic.argmin() / len(n_components_range))
        )
        plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
        spl.set_xlabel(str(n_feature) + " features Number of components")
        spl.legend([b[0] for b in bars], cv_types)

        splt = plt.subplot(2, 1, 2)
        print("model: ", model_likelihood_list)
        plt.title("model likelihood")
        plt.xticks(n_components_range)
        plt.plot(n_components_range, model_likelihood_list)
    # if(if_plot[3] == True):
    #     write_cluster(csv_path, best_cluster_labels)
    plt.show()
    # print("best cluster numbers = ", best_cluster_num, ", value = ", clf, ", best method = ", best_method, 'lowest_bic: ', lowest_bic)
    density = best_gmm.predict_proba(X)
    # print(best_gmm.weights_)
    return bic, [best_component, best_gmm, lowest_bic]

# all the data is clustered
def gmm_t_no_split(data, if_plot, instrument_list, n_feature):
    lowest_bic = np.infty
    bic = []
    n_components_range = range(2, 11)
    if(if_plot[0]):
      cv_types = ["tied", "diag", "full"]
      # cv_types = ["diag", "full"]
    else:
        # cv_types = ["full"]
        cv_types = ["full"]

    for cv_type in cv_types:
        silhouette_score_set = []
        model_likelihood_list = []
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type=cv_type, max_iter = 200, tol = 7e-4
            )
            gmm.fit(data)
            bic.append(gmm.bic(data))
            cluster_labels = gmm.predict(data)
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_component = n_components
                best_method = cv_type
                best_cluster_labels = cluster_labels
            model_likelihood = gmm.score(data)
            sample_likelihood = gmm.score_samples(data)
            model_likelihood_list.append(model_likelihood)
        if (if_plot[4]):
            plot_cluster_number_dis(best_component, instrument_list, best_cluster_labels, n_feature, cv_type)
            # print("best cluster labels: ", best_cluster_labels)

    bic = np.array(bic)
    # for i in range(0,n_cluster-2):
    #   print(i+2, "clusters: ", bic[i])
    color_iter = itertools.cycle(["navy", "turquoise", "cornflowerblue", "darkorange"])
    clf = best_gmm
    best_cluster_num = best_component
    bars = []


    if(if_plot[1]):
        plt.figure(figsize=(8, 6))
        spl = plt.subplot(2, 1, 1)
        if(if_plot[0]):
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                xpos = np.array(n_components_range) + 0.2 * (i - 2)
                bars.append(
                    plt.bar(
                        xpos,
                        bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
                        width=0.2,
                        color=color,
                    )
                )
        if not (if_plot[0]):
            for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
                    xpos = np.array(n_components_range)
                    bars.append(
                        plt.bar(
                            xpos,
                            bic[i * len(n_components_range): (i + 1) * len(n_components_range)],
                            width=0.2,
                            color=color,
                        )
                    )
        plt.xticks(n_components_range)
        plt.ylim([bic.min() * 1.01 - 0.01 * bic.max(), bic.max()])
        plt.title(str(n_feature) + " features BIC score per model")
        xpos = (
                np.mod(bic.argmin(), len(n_components_range))
                + 0.65
                + 0.2 * np.floor(bic.argmin() / len(n_components_range))
        )
        plt.text(xpos, bic.min() * 0.97 + 0.03 * bic.max(), "*", fontsize=14)
        spl.set_xlabel(str(n_feature) + " features Number of components")
        spl.legend([b[0] for b in bars], cv_types)

        splt = plt.subplot(2, 1, 2)
        print("model: ", model_likelihood_list)
        plt.title("model likelihood")
        plt.xticks(n_components_range)
        plt.plot(n_components_range, model_likelihood_list)
    # if(if_plot[3] == True):
    #     write_cluster(csv_path, best_cluster_labels)
    plt.show()
    # print("best cluster numbers = ", best_cluster_num, ", value = ", clf, ", best method = ", best_method, 'lowest_bic: ', lowest_bic)
    density = best_gmm.predict_proba(data)
    # print(best_gmm.weights_)
    return bic, [best_component, best_gmm, lowest_bic]

def gmm_no_plot(total_data, train_set, test_set, cluster_num, iter):
    for i in range(iter):
        lowest_bic = np.infty
        bic = []
        # cv_types = "full"
        cv_types = "diag"
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=cluster_num, covariance_type=cv_types, max_iter=300, tol=7e-4
        )
        gmm.fit(train_set)
        bic.append(gmm.bic(test_set))
        if(bic[-1] < lowest_bic):
            lowest_bic = bic[-1]
            best_gmm = gmm
    return lowest_bic, best_gmm


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def hiera_cluster(data):
    linkage = ['ward', 'complete', 'average', 'single']
    for ele in linkage:
        hc = sklearn.cluster.AgglomerativeClustering(distance_threshold = 0,linkage=ele, n_clusters=None)
        hc.fit(data)
        fig,ax = plt.subplots()
        plt.title(ele + ", Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(hc, truncate_mode="level", p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()

def gmm_cluster(data, if_plot, instrument, n_feature, whether_split):
    train_data, test_data = split_train_test(data)
    write_fpath = r"C:\Users\xrw\Desktop"
    csv_path1 = os.path.join(write_fpath, "4_14_" + str(n_feature) + "_6cluster" + '.csv')
    best_bic = np.inf
    for i in range(50):
        print("attempt: ", i)
        if(whether_split):
            bic, best = gmm_t(data,train_data,test_data, if_plot, instrument,n_feature)
            print("best bic: ", bic)
        else:
            bic, best = gmm_t_no_split(data, if_plot, instrument, n_feature)
        if (best[2] < best_bic):
            best_bic = best[2]
            best_gmm = best[1]
            best_component = best[0]
    pre_label = best_gmm.predict(data)
    plot_cluster_number_dis(best_component, instrument, pre_label, n_feature)
    plt.savefig(r"C:\Users\xrw\Desktop\distribution")
    col = ['avg_zcr', 'total_energy','pitch ratio2/1', 'pitch ratio3/1', 'pitch ratio sp/1','harmonic energy ratio1', 'harmonic energy ratio2', 'harmonic energy ratio3', 'harmonic energy ratio sp']
    data = pd.DataFrame(data)
    data['label'] = pre_label
    # sns.pairplot(data,hue = 'label',diag_kind="kde")
    # plt.savefig(r"C:\Users\邢\Desktop\data_dis")
    plt.show()
    print("best bic: ", best_bic, "best comoponent: ", best_component)
    if(if_plot[3]):
        write_cluster(csv_path1, pre_label, n_feature)


def gmm10(use_data,train_data,test_data, if_plot, instrument, times, n_feature):
    best_bic = np.inf
    for i in range(times):
        bic, best = gmm_t(use_data,train_data,test_data, if_plot, instrument,n_feature)
        if (best[2] < best_bic):
            best_bic = best[2]
            best_gmm = best[1]
    return best_bic, best_gmm

def g_cluster(data, instrument_list, if_plot, n_feature, whether_split):
    gmm_cluster(data, if_plot, instrument_list, n_feature, whether_split)

if __name__ == '__main__':
    n_cluster = 11
    nt_feature = 60
    dataset = pd.read_csv(csv_path)
    data_columns = dataset.columns
    dataset = dataset.values
    option_list = ['show_four_split', 'show_four_no_split', '20iteration_write cluster']

    print("data: ", dataset.shape)
    if not (hie_or_gmm):
        # 1.four or one method  2. whether draw the picture of every iteration 3. whether draw the average picture 4. whether write cluster number 5.whether show instrument list
        # One method
        if_plot = [False, False, False, False, False]
        # four method
        # if_plot = [True, True, False, False, True]
        # one method, average picture
        # if_plot = [False, False, True, False, True]
        if (dataset.shape[1] == nt_feature + 1):
            data = dataset[:, :-1]
            instrument_list = dataset[:, -1]
            data_columns = data_columns
        elif (dataset.shape[1] == nt_feature + 2):
            data = dataset[:, :-2]
            instrument_list = dataset[:, -2]
            data_columns = data_columns[:-1]
        instrument_list = np.squeeze(instrument_list)
        MR_data = Standardization.MR_scale(data)
        MVR_data = Standardization.MVR_scale(data)
        Log_data = Standardization.Log_scale(data)
        RR_data = Standardization.RR_scale(data)

        # choose standization method
        target_data = Log_data

        # gmm50(MR_data, [False,False,False,False,False], instument_list)
        # test_data = np.delete(MR_data, (1,3,5,7,9,11,13,15,17,19,24,25,26,27), axis=1)
        # test_data = np.delete(MR_data,(19,20,21,22,23,24,25,26,27),axis=1)
        # test_data = np.delete(test_data, (0,1,3,5,7,9), axis=1)
        # test_data = np.delete(MR_data,(22,28,50,51,52,53,54), axis=1)
        # delete 4th and 5th harmonic
        except_5th_harmonic = np.delete(target_data, (11, 17, 22, 28, 50, 51, 52, 53, 54), axis=1)
        n_feature1 = 51
        except_4and5th_harmonic = np.delete(target_data,
                                            (10, 11, 16, 17, 21, 22, 27, 28, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54),
                                            axis=1)
        n_feature2 = 42
        # delete 1/5 parts
        except_5parts_harmonic = np.delete(target_data, np.arange(30, 60, 1), axis=1)
        n_feature3 = 30
        except_All5parts = np.delete(except_5parts_harmonic, (3, 4, 5, 6, 7), axis=1)
        n_feature4 = 25

        except_5th_5parts_harmonic = np.delete(except_5parts_harmonic, (11, 17, 22, 28), axis=1)
        n_feature5 = 26
        except_4and5th_5parts_harmonic = np.delete(except_5parts_harmonic, (10, 11, 16, 17, 21, 22, 27, 28), axis=1)
        n_feature6 = 22

        except_5th_All5parts = np.delete(except_5th_5parts_harmonic, (3, 4, 5, 6, 7), axis=1)
        n_feature7 = 21
        except_4and5th_All5parts = np.delete(except_4and5th_5parts_harmonic, (3, 4, 5, 6, 7), axis=1)
        n_feature8 = 17

        # delete attack and 1/5 parts
        except_attack_5parts_harmonic = np.delete(except_5parts_harmonic, np.arange(19, 30, 1), axis=1)
        n_feature9 = 19
        except_attack_All5parts = np.delete(except_attack_5parts_harmonic, (3, 4, 5, 6, 7), axis=1)
        n_feature10 = 14
        except_5th_attack_5parts_harmonic = np.delete(except_attack_5parts_harmonic, (11, 17), axis=1)
        n_feature11 = 17
        except_5th_attack_All5parts = np.delete(except_attack_5parts_harmonic, (3, 4, 5, 6, 7, 11, 17), axis=1)
        n_feature12 = 12
        except_4and5th_attack_5parts_harmonic = np.delete(except_attack_5parts_harmonic, (10, 11, 16, 17), axis=1)
        n_feature13 = 15
        except_4and5th_attack_All5parts = np.delete(except_attack_5parts_harmonic, (3, 4, 5, 6, 7, 10, 11, 16, 17),
                                                    axis=1)
        n_feature14 = 10

        # delete all attack and all 1/5 parts
        except_Allattack_All5parts = np.delete(except_attack_All5parts, (2), axis=1)
        n_feature15 = 13
        except_5th_Allattack_All5parts = np.delete(except_5th_attack_All5parts, (2), axis=1)
        n_feature16 = 11
        except_4and5th_Allattack_All5parts = np.delete(except_4and5th_attack_All5parts, (2), axis=1)
        n_feature17 = 9
        except_absolute = np.delete(target_data, np.arange(0, 8, 1), axis=1)
        except_absolute_la = np.delete(data_columns, np.arange(0, 8, 1))
        n_feature4 = 52
        except_5th_harmonic = np.delete(target_data, (11, 17, 22, 28, 50, 51, 52, 53, 54), axis=1)
        n_feature1 = 51
        except_4and5th_harmonic = np.delete(target_data,
                                            (10, 11, 16, 17, 21, 22, 27, 28, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54),
                                            axis=1)
        n_feature2 = 42

        except_absolute_5th_harmonic = np.delete(except_absolute, (3, 9, 14, 20, 42, 43, 44, 45, 46), axis=1)
        n_feature18 = 43
        except_absolute_4and5th_harmonic = np.delete(except_absolute,
                                                     (2, 3, 8, 9, 13, 14, 19, 20, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                                      46),
                                                     axis=1)
        except_absolute_4and5th_harmonic_la = np.delete(except_absolute_la,
                                                        (2, 3, 8, 9, 13, 14, 19, 20, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                                                         46))
        n_feature19 = 34

        print(except_absolute_4and5th_harmonic_la)
        # except_absolute_4and5th_harmonic_filtered = remove_alloutliers_m(except_absolute_4and5th_harmonic, True, except_absolute_4and5th_harmonic_la)
        #
        # print(except_absolute_4and5th_harmonic_filtered)

        # gmm_cluster(MR_data, if_plot, instument_list, nt_feature)
        # gmm_cluster(MR_data, if_plot, instument_list, nt_feature)
        # gmm_cluster(except_4and5th_harmonic, if_plot, instument_list, n_feature2)
        # gmm_cluster(except_absolute,if_plot, instument_list, n_feature4)
        # gmm_cluster(except_absolute, if_plot, instument_list, n_feature4)
        option = option_list[2]
        total_data = except_absolute_5th_harmonic
        print("column: ", total_data.shape[1])
        t_feature = n_feature14
        if(option == "show_four_split"):
            if_plot = [True, True, False, False, True]
            train_data, test_data = split_train_test(total_data)
            gmm_t(total_data, train_data, test_data, if_plot, instrument_list, t_feature)
        elif(option == 'show_four_no_split'):
            if_plot = [True, True, False, False, True]
            gmm_t_no_split(total_data, if_plot, instrument_list, t_feature)
        elif(option == '20iteration_write cluster'):
            # write cluster
            if_plot = [False, False, False, True, False]
            g_cluster(except_4and5th_attack_All5parts, instrument_list, if_plot, t_feature, whether_split = 1)

        # train_data, test_data = split_train_test(except_absolute_5th_harmonic)
        # gmm_t()
        # gmm_cluster(MVR_data, if_plot, instument_list)
        # best_bic, best_gmm = gmm10(MVR_data, if_plot, instument_list,10)
        # gmm_cluster(Log_data, if_plot, instument_list)
        # gmm_cluster(RR_data, if_plot, instument_list)
    else:
        if (dataset.shape[1] == nt_feature + 1):
            data = dataset[:, :-1]
            instrument_list = dataset[:, -1]
            data_columns = data_columns[:-1]
        elif (dataset.shape[1] == nt_feature + 2):
            data = dataset[:, :-2]
            instrument_list = dataset[:, -2]
            data_columns = data_columns[:-2]

        MR_data = Standardization.MR_scale(data)
        MVR_data = Standardization.MVR_scale(data)
        log_data = Standardization.Log_scale(data)
        print(log_data.shape)
        # MR_data = np.delete(MR_data, 297, axis = 0)
        # adata = np.delete(MR_data, (221,523,755), axis=0)
        # hiera_cluster(adata)
        # MVR_data = np.delete(MVR_data, (221, 523, 586, 755), axis=0)
        # hiera_cluster(MVR_data)
        # hiera_cluster(MR_data)
        for i in range(log_data.shape[0]):
            print(log_data[i])
        # print(log_data)
        hiera_cluster(log_data)
        n_da = np.delete(dataset, (221, 523, 586, 755), axis=0)

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import Standardization
import clustering

n_feature = 28
# csv_path = r"C:\Users\邢\Desktop\10 nguru recording\totaldataset_18_t.csv"
# csv_path = r"C:\Users\邢\Desktop\10 nguru recording\5note_dataset_18_t.csv"
csv_path = r"C:\Users\邢\Desktop\10 nguru recording\finalset_t.csv"


def get_likelihood(x_i,mu,sigma):
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)   # natural logarithm of determinant of matrix
    y = np.linalg.solve(sigma, x_i - mu)   # solve the function AX=b
    c = np.dot(x_i - mu, y)
    return -0.5*(a + b + c)

def get_likelihood(x_i,mu,sigma):
    n = len(mu)
    a = n * np.log(2 * np.pi)
    _, b = np.linalg.slogdet(sigma)   # natural logarithm of determinant of matrix
    y = np.linalg.solve(sigma, x_i - mu)   # solve the function AX=b
    c = np.dot(x_i - mu, y)
    return -0.5*(a + b + c)

def show(data,instrument, times):
    for i in range(n_feature):
        new_data = np.delete(data, i, axis=1)
        best_bic, best_gmm = clustering.gmm10(new_data, if_plot, instrument, times)
        joblib.dump(best_gmm, 'bestgmm' + str(i) + '.pkl')

def get_prob(data):
    gmm = joblib.load('bestgmm/bestgmm.pkl')
    best_cluster_labels = gmm.predict(data)
    print(best_cluster_labels)
    best_component = gmm.n_components
    best_init = gmm.init_params
    density = gmm.predict_proba(data)
    likelihood = gmm.score_samples(data)
    a = gmm.weights_
    covariance = gmm.covariances_
    means = gmm.means_
    param = gmm.get_params()
    print(best_component)
    print("weight:", a)
    print("likelihood:", likelihood.shape)
    print(likelihood)
    print("density: ", density.shape)
    print(density)
    print("covariance: ", covariance.shape)
    print(covariance)
    print("mean:", means.shape)
    print(means)
    print(param)
    per_s_c_likelihood = []
    posterior = []

    for i in range(data.shape[0]):
        xi_likelihood = []
        for j in range(best_component):
            sam_likelihood = get_likelihood(data[i], means[j], covariance[j])
            xi_likelihood.append(sam_likelihood)
        per_s_c_likelihood.append(xi_likelihood)
    per_s_c_likelihood = np.array(per_s_c_likelihood)
    print("per sample/cluster likelihood: ", per_s_c_likelihood.shape)
    # per_s_c_likelihood = numpy.power(10, per_s_c_likelihood)
    print("per likelihood: ", per_s_c_likelihood)
    pd.DataFrame(per_s_c_likelihood).to_csv(r"C:\Users\邢\Desktop\likelihood.csv")

    for i in range(data.shape[0]):
        sam_density = density[i]
        sam_likelihood = per_s_c_likelihood[i]
        for j in range(best_component):
            joint_prob = np.multiply(sam_density, sam_likelihood)
            sam_posterior = joint_prob / np.sum(joint_prob)
        posterior.append(sam_posterior)
    posterior = np.array(posterior)
    print("posterior: ", posterior.shape)
    print(posterior)
    # pd.DataFrame(posterior).to_csv(r"C:\Users\邢\Desktop\posterior.csv")
    # print(
    #     "\nFor n_clusters =",
    #     n_components,
    #     "The average silhouette_score is :",
    #     silhouette_avg,
    # )
    ins_cluster = []
    instrument_num = 10
    instrument_l = []
    for i in range(instrument_num):
        instrument_l.append("instrument " + str(instrument_num))
    cluster_iter = 0
    for i in range(best_component):
        in_list = []
        for j in range(instrument_list.shape[0]):
            if (best_cluster_labels[j] == cluster_iter):
                in_list.append(instrument_list[j])
        cluster_iter += 1
        ins_cluster.append(in_list)
    plt.subplot(2, 1, 1)
    plt.xticks(range(1, instrument_num + 1))
    plt.legend(loc='upper right')
    plt.hist(ins_cluster, bins=10, edgecolor="r", alpha=0.5, label=instrument_l)
    plt.title("instrument cluster result")
    #     best_component_list.append("cluster" + str(i))
    # for i in range(len(instrument_l)):

    if (if_plot[3] == False):
        clustering.write_cluster(csv_path, best_cluster_labels)

    # print(ins_cluster)
    plt.subplot(2, 1, 2)
    plt.xticks(range(0, best_component))
    plt.hist(best_cluster_labels, bins=10)
    plt.title("cluster result")
    plt.show()
    print(gmm.bic(data))


if __name__ == '__main__':
    dataset = np.loadtxt(csv_path, delimiter=",")
    # 1.four or one method  2. whether draw the picture 3. whether draw the average picture 4. whether write cluster number5.whether show instrument list
    # One method
    if_plot = [False, True, False, True, False]
    # four method
    # if_plot = [True, True, False, False]
    # one method, average picture
    # if_plot = [False, False, True, False]
    print(dataset.shape)
    if (dataset.shape[1] == n_feature + 1):
        data = dataset[:, :-1]
        instrument_list = dataset[:, -1]
    elif (dataset.shape[1] == n_feature + 2):
        data = dataset[:, :-2]
        instrument_list = dataset[:, -2]
    print(data.shape)
    instrument_list = np.squeeze(instrument_list)
    data = Standardization.MVR_scale(data)
    MR_data = Standardization.MR_scale(data)
    MVR_data = Standardization.MVR_scale(data)
    # show(MVR_data,instrument_list,10)
    get_prob(data)
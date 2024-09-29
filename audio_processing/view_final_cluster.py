import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,
                       cluster_std=0.60, random_state=0)
X = X[:, ::-1] #交换列是为了方便画图

def get_cluster_num(c):
    dict = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0}
    for i in range(50):
        if (int(c[i][0]) == 2):
            dict['2'] += 1
        elif (int(c[i][0]) == 3):
            dict['3'] += 1
        elif (int(c[i][0]) == 4):
            dict['4'] += 1
        elif (int(c[i][0]) == 5):
            dict['5'] += 1
        elif (int(c[i][0]) == 6):
            dict['6'] += 1
        elif (int(c[i][0]) == 7):
            dict['7'] += 1
        elif (int(c[i][0]) == 8):
            dict['8'] += 1
        elif (int(c[i][0]) == 9):
            dict['9'] += 1
        elif (int(c[i][0]) == 10):
            dict['10'] += 1
    max = 0
    max_c = ''
    for i in range(9):
        if(dict[str(i+2)] > max):
            max = dict[str(i+2)]
            max_c = str(i+2)
    print(dict)
    print("cluster: " + max_c + ", max number: ", max)

def show_p_and_c(n):
    p = np.load('p' + str(n) + 'b.npy')
    c = np.load('c' + str(n) + 'b.npy')
    print("\ncluster distribution: ")
    print(str(n) + 'th data set: \n', np.average(p))
    get_cluster_num(c)
    return np.average(p)


n_feature_set = 39
dataset_list = np.array(["LOG", "except_5th_harmonic", "except_4and5th_harmonic", "except_5parts_harmonic", "except_All5parts",
                "except_5th_5parts_harmonic", "except_4and5th_5parts_harmonic", "except_5th_All5parts",
                "except_4and5th_All5parts", "except_attack_5parts_harmonic", "except_attack_All5parts",
                "except_5th_attack_5parts_harmonic", "except_5th_attack_All5parts",
                "except_4and5th_attack_5parts_harmonic", "except_4and5th_attack_All5parts",
                "except_Allattack_All5parts", "except_5th_Allattack_All5parts", "except_4and5th_Allattack_All5parts", "except_absolute", 'except_absolute_5th', "except_absolute_4and5th", "except_absolute_5parts_harmonic", "except_absolute_5th_5parts_harmonic",
                "except_absolute_4and5th_5parts_harmonic", "except_absolute_attack_5parts_harmonic", "except_absolute_5th_attack_5parts_harmonic", "except_absolute_4and5th_attack_5parts_harmonic",'except_zcr', 'except_zcr_5th_harmonic', 'except_zcr_4and5th_harmonic', 'except_zcr_5parts_harmonic',
                'except_sp', 'except_sp_5th_harmonic', 'except_sp_4and5th_harmonic', 'except_sp_5parts_harmonic', 'except_sp_attack_5parts_harmonic',  'except_sp_attack_All5parts', 'except_sp_4and5th_attack_All5parts', 'except_sp_4and5th_Allattack_All5parts'])
all_element = np.array(['zcr', 'total_energy', 'attack_energy', '5_part_energy', 'three_harmonic', '4_harmonic', '5_harmonic', 'sp_harmonic', 'three_attack_harmonic', '4_attack_harmonic', '5_attack_harmonic', 'sp_attack_harmonic', '5part_three_harmonic', '5part_4_harmonic', '5part_5_harmonic', '5part_sp_harmonic'])
element_list = []
element1 = all_element
element2 = np.delete(all_element, [6,10,14])
element3 = np.delete(all_element, [5,6,9,10,13,14])
element4 = np.delete(all_element, [12,13,14,15])
element5 = np.delete(all_element, [3,12,13,14,15])
element6 = np.delete(all_element, [6,10,12,13,14,15])
element7 = np.delete(all_element, [5,6,9,10,12,13,14,15])
element8 = np.delete(all_element, [3,6,10,12,13,14,15])
element9 = np.delete(all_element, [3,5,6,9,10,12,13,14,15])
element10 = np.delete(all_element, [8,9,10,11,12,13,14,15])
element11 = np.delete(all_element, [3,8,9,10,11,12,13,14,15])
element12 = np.delete(all_element, [6,8,9,10,11,12,13,14,15])
element13 = np.delete(all_element, [3,6,8,9,10,11,12,13,14,15])
element14 = np.delete(all_element, [5,6,8,9,10,11,12,13,14,15])
element15 = np.delete(all_element, [3,5,6,8,9,10,11,12,13,14,15])
element16 = np.delete(all_element, [2,3,8,9,10,11,12,13,14,15])
element17 = np.delete(all_element, [2,3,6,8,9,10,11,12,13,14,15])
element18 = np.delete(all_element, [2,3,5,6,8,9,10,11,12,13,14,15])
element19 = np.delete(all_element, [0,1,2,3])
element20 = np.delete(all_element, [0,1,2,3,6,10,14])
element21 = np.delete(all_element, [0,1,2,3,5,6,9,10,13,14])
element22 = np.delete(all_element, [0,1,2,3,12,13,14,15])
element23 = np.delete(all_element, [0,1,2,3,6,10,12,13,14,15])
element24 = np.delete(all_element, [0,1,2,3,5,6,9,10,12,13,14,15])
element25 = np.delete(all_element, [0,1,2,3,8,9,10,11,12,13,14,15])
element26 = np.delete(all_element, [0,1,2,3,6,8,9,10,11,12,13,14,15])
element27 = np.delete(all_element, [0,1,2,3,5,6,8,9,10,11,12,13,14,15])
element28 = np.delete(all_element, [0])
element29 = np.delete(all_element, [0,6,10,14])
element30 = np.delete(all_element, [0,5,6,9,10,13,14])
element31 = np.delete(all_element, [0,12,13,14,15])
element32 = np.delete(all_element, [7,11,15])
element33 = np.delete(all_element, [6,7,10,11,14,15])
element34 = np.delete(all_element, [5,6,7,9,10,11,13,14,15])
element35 = np.delete(all_element, [7,11,12,13,14,15])
element36 = np.delete(all_element, [7,8,9,10,11,12,13,14,15])
element37 = np.delete(all_element, [3,7,8,9,10,11,12,13,14,15])
element38 = np.delete(all_element, [3,5,6,7,8,9,10,11,12,13,14,15])
element39 = np.delete(all_element, [2,3,5,6,7,8,9,10,11,12,13,14,15])

element_list.append(element1)
element_list.append(element2)
element_list.append(element3)
element_list.append(element4)
element_list.append(element5)
element_list.append(element6)
element_list.append(element7)
element_list.append(element8)
element_list.append(element9)
element_list.append(element10)
element_list.append(element11)
element_list.append(element12)
element_list.append(element13)
element_list.append(element14)
element_list.append(element15)
element_list.append(element16)
element_list.append(element17)
element_list.append(element18)
element_list.append(element19)
element_list.append(element20)
element_list.append(element21)
element_list.append(element22)
element_list.append(element23)
element_list.append(element24)
element_list.append(element25)
element_list.append(element26)
element_list.append(element27)
element_list.append(element28)
element_list.append(element29)
element_list.append(element30)
element_list.append(element31)
element_list.append(element32)
element_list.append(element33)
element_list.append(element34)
element_list.append(element35)
element_list.append(element36)
element_list.append(element37)
element_list.append(element38)
element_list.append(element39)
print("element_list: ", element_list[0])
print(np.setdiff1d(element_list[0], element_list[1]))
print((np.setdiff1d(element_list[0], element_list[1]) == ['5_attack_harmonic', '5_harmonic', '5part_5_harmonic']).all())
def get_differ(target, length):
    diff_list = []
    for i in range(n_feature_set):
        for j in range(i+1, n_feature_set, 1):
            diffa = np.setdiff1d(element_list[i],element_list[j],assume_unique=True)
            lenth_diff = len(element_list[i]) - len(element_list[j])
            if(lenth_diff == length and len(diffa) == length):
                if ((diffa == target).all()):
                    diff_list.append([i, j])
    return diff_list

def get_prob_diff(diff_list):
    prob_list = []
    for ele in diff_list:
        print(ele)
        prob1 = total_prob[ele[0]]
        prob2 = total_prob[ele[1]]
        prob = prob1 - prob2
        prob_list.extend(prob)
    prob_list = np.around(prob_list,5)
    print(prob_list)
    return prob_list

ex_5th = [0,1,5,7,11,12,16]
ex_4and5th_toge = [0,1,2,5,6,7,8,11,12,13,14,16,17]
ex_4and5th_only = [0,2,6,8,13,14,17]
ex_attack = [0,9,10,11,12,13,14,15,16,17]
ex_All = [0,4,7,8,10,12,14,15,16,17]

determine_attack = []

print(len(dataset_list))

def show_certain_p_and_c(path):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    # plt.figure()
    for i in range(18):
        name = dataset_list[i]
        prob = np.load(os.path.join("../result_data/certain_cluster", name + " certain_prob_list.npy"))
        prob_distribution = np.load(os.path.join("../result_data/certain_cluster", name + " certain_prob_distribution_list.npy"))
        print(prob)
        print(prob_distribution)
        all_prob_list.append(prob)
        plt.plot(cluster_num, prob, label = name)
    plt.legend(loc='upper right')
    plt.title("50")
    plt.show()
    all_prob_list = np.array(all_prob_list)

# get corresponding result according to 100traince, with n-2 cluster number
def show_certain_p_and_c2(n):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    set_num = np.arange(1, 19, 1)
    # plt.figure()
    for i in range(18):
        name = dataset_list[i]
        prob = np.load(name + " certain_prob_list.npy")
        prob_distribution = np.load(name + " certain_prob_distribution_list.npy")
        print(prob)
        print(prob_distribution)
        all_prob_list.append(prob[n-2])
    all_prob_list = np.array(all_prob_list)
    plt.xticks(set_num)
    plt.title("100")
    plt.legend(loc='upper right')
    plt.plot(set_num, all_prob_list, label=name)
    plt.show()

# get path mean result
def show_different_feature(path = None):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    plt.figure(figsize=(10, 10))
    for i in range(19):
            name = dataset_list[i]
            prob_path = os.path.join(path, name + " certain_prob_list.npy")
            if(os.path.exists(prob_path)):
                prob = np.load(prob_path)
                prob_distribution = np.load(os.path.join(path, name + " certain_prob_distribution_list.npy"))
                # print(prob_distribution[:, :50])
                # print(prob)
                # print(prob_distribution.shape)
                all_prob_list.append(prob)
                plt.plot(cluster_num, prob, label=name)
    plt.legend(loc='upper right')
    plt.title(path + "total")
    plt.show()

# get path mean result
def show_different_feature_certain_cluster_num(cluster_c, path = ''):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    plt.figure(figsize=(10, 10))
    for i in range(n_feature_set):
            name = dataset_list[i]
            prob_path = os.path.join(path, name + " certain_prob_list.npy")
            if(os.path.exists(prob_path)):
                prob = np.load(prob_path)
                all_prob_list.append(prob[cluster_c - 2])
    return all_prob_list

def show_different_feature_multiplylog(path=None):
    all_prob_list = []
    cluster_num = np.arange(2, 11, 1)
    plt.figure(figsize=(10, 10))
    log_List = [np.log(2)/2, np.log(3)/3, np.log(4)/4, np.log(5)/5, np.log(6)/6, np.log(7)/7, np.log(8)/8, np.log(9)/9, np.log(10)/10]
    for i in range(20):
        name = dataset_list[i]
        prob_path = os.path.join(path, name + " certain_prob_list.npy")
        if (os.path.exists(prob_path)):
            prob = np.load(prob_path)
            prob = prob * log_List
            prob_distribution = np.load(os.path.join(path, name + " certain_prob_distribution_list.npy"))
            print(prob_distribution[:, :50])
            print(prob)
            print(prob_distribution.shape)
            all_prob_list.append(prob)
            plt.plot(cluster_num, prob, label=name)
    plt.legend(loc='upper right')
    plt.title(path + "total")

    # plt.figure(figsize=(10, 10))
    # for i in range(19):
    #     if i in ex_attack:
    #         name = dataset_list[i]
    #         prob = np.load(os.path.join(path, name + " certain_prob_list.npy"))
    #         prob_distribution = np.load(os.path.join(path, name + " certain_prob_distribution_list.npy"))
    #         print(prob_distribution[:, :50])
    #         print(prob)
    #         print(prob_distribution.shape)
    #         all_prob_list.append(prob)
    #         plt.plot(cluster_num, prob, label=name)
    # plt.legend(loc='upper right')
    # plt.title(path + "attack")
    #
    # plt.figure(figsize=(10, 10))
    # for i in range(19):
    #     if i in [0,18]:
    #         name = dataset_list[i]
    #         prob = np.load(os.path.join(path, name + " certain_prob_list.npy"))
    #         print(prob_distribution[:, :50])
    #         print(prob)
    #         print(prob_distribution.shape)
    #         all_prob_list.append(prob)
    #         plt.plot(cluster_num, prob, label=name)
    # plt.legend(loc='upper right')
    # plt.title(path + "ex_absolute")

    plt.show()
    all_prob_list = np.array(all_prob_list)

# get path medianresult
def show_median_different_feature(path = None):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    plt.figure(figsize=(10, 10))
    for i in range(19):
            name = dataset_list[i]
            prob = np.load(os.path.join(path, name + " certain_prob_list.npy"))
            prob_distribution = np.load(os.path.join(path, name + " certain_prob_distribution_list.npy"))

            median_prob = np.median(prob_distribution[:,:50], axis = 1)
            print(prob_distribution[:, :50])
            print("ss:", median_prob)
            print(prob)
            print(prob_distribution.shape)
            all_prob_list.append(median_prob)
            plt.plot(cluster_num, median_prob, label=name)
    plt.legend(loc='upper right')
    plt.title(path + "total")

    plt.figure(figsize=(10, 10))
    for i in range(19):
        if i in ex_attack:
            name = dataset_list[i]
            prob = np.load(os.path.join(path, name + " certain_prob_list.npy"))
            prob_distribution = np.load(os.path.join(path, name + " certain_prob_distribution_list.npy"))
            median_prob = np.median(prob_distribution[:,:50], axis = 1)
            print(prob)
            print(prob_distribution.shape)
            all_prob_list.append(median_prob)
            plt.plot(cluster_num, median_prob, label=name)
    plt.legend(loc='upper right')
    plt.title(path + "attack")

    plt.figure(figsize=(10, 10))
    for i in range(19):
        if i in [0,18]:
            name = dataset_list[i]
            prob = np.load(os.path.join(path, name + " certain_prob_list.npy"))
            median_prob = np.median(prob_distribution[:,:50], axis = 1)
            print(prob)
            print(prob_distribution.shape)
            all_prob_list.append(median_prob)
            plt.plot(cluster_num, median_prob, label=name)
    plt.legend(loc='upper right')
    plt.title(path + "ex_absolute")

    plt.show()
    all_prob_list = np.array(all_prob_list)

# get corresponding result according to 100traince
def show_different_feature2(path):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    plt.figure(figsize=(10, 10))
    for i in range(18):
        name = dataset_list[i]
        prob = np.load(name + " certain_prob_list.npy")
        prob_distribution = np.load(name + " certain_prob_distribution_list.npy")
        print(prob)
        print(prob_distribution)
        all_prob_list.append(prob)
        plt.plot(cluster_num, prob, label=name)
    plt.legend(loc='upper right')
    plt.title("100")

    plt.figure(figsize=(10,10))
    for i in range(18):
        if i in ex_4and5th_toge:
            name = dataset_list[i]
            prob = np.load(name + " certain_prob_list.npy")
            prob_distribution = np.load(name + " certain_prob_distribution_list.npy")
            print(prob)
            print(prob_distribution)
            all_prob_list.append(prob)
            plt.plot(cluster_num, prob, label=name)
    plt.legend(loc='upper right')
    plt.title("100 4and5th")

    plt.figure(figsize=(10,10))
    for i in range(18):
        if i in ex_attack:
            name = dataset_list[i]
            prob = np.load(name + " certain_prob_list.npy")
            prob_distribution = np.load(name + " certain_prob_distribution_list.npy")
            print(prob)
            print(prob_distribution)
            all_prob_list.append(prob)
            plt.plot(cluster_num, prob, label=name)
    plt.legend(loc='upper right')
    plt.title("100 attack")

    plt.figure(figsize=(10,10))
    for i in range(18):
        if i in ex_All:
            name = dataset_list[i]
            prob = np.load(name + " certain_prob_list.npy")
            prob_distribution = np.load(name + " certain_prob_distribution_list.npy")
            print(prob)
            print(prob_distribution)
            all_prob_list.append(prob)
            plt.plot(cluster_num, prob, label=name)

    plt.legend(loc='upper right')
    plt.title("100 All")

    plt.show()
    all_prob_list = np.array(all_prob_list)

def plot_prob_result(n, path, choose_index):
    all_prob_list = []
    set_num = 1
    plt.figure()
    for i in range(18):
        if i in choose_index:
            name = dataset_list[i]
            prob_path = os.path.join(path, name + " certain_prob_list.npy")
            if (os.path.exists(prob_path)):
                prob = np.load(prob_path)
                prob_distribution = np.load(os.path.join(path, name + " certain_prob_distribution_list.npy"))
                print(prob)
                print(prob_distribution)
                all_prob_list.append(prob)
                plt.plot(set_num, prob[:, n - 2], label=name)
    plt.legend(loc='upper right')
    plt.title("100 5th")

# certain_cluster, get corresponding result
def show_different_feature_certain_cluster(n):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    set_num = 1
    plot_prob_result(n, '', ex_5th)
    plt.legend(loc='upper right')
    plt.title("100 5th")

    plt.figure()
    for i in range(18):
        if i in ex_attack:
            name = dataset_list[i]
            prob = np.load(name + " certain_prob_list.npy")
            prob_distribution = np.load(name + " certain_prob_distribution_list.npy")
            print(prob)
            print(prob_distribution)
            all_prob_list.append(prob)
            plt.plot(set_num, prob[:,n-2], label=name)
    plt.legend(loc='upper right')
    plt.title("100 attack")

    plt.figure()
    for i in range(18):
        if i in ex_All:
            name = dataset_list[i]
            prob = np.load(name + " certain_prob_list.npy")
            prob_distribution = np.load(name + " certain_prob_distribution_list.npy")
            print(prob)
            print(prob_distribution)
            all_prob_list.append(prob)
            plt.plot(set_num, prob[:,n-2], label=name)

    plt.legend(loc='upper right')
    plt.title("100 All")

    plt.show()
    all_prob_list = np.array(all_prob_list)

def show_bic(path = None):
    all_prob_list = []
    cluster_num = np.arange(2,11,1)
    total_bic_list = []
    print("len: ", len(dataset_list))
    for i in range(n_feature_set):
            name = dataset_list[i]
            bic_path = os.path.join(path, name + " bic_list.npy")
            if(os.path.exists(bic_path)):
                bic_list = np.load(bic_path)
                print(name + " bic: ")
                print(bic_list)
                total_bic_list.append(bic_list)
    total_bic_list = np.around(np.array(total_bic_list),2)
    bic_dataframe = pd.DataFrame(total_bic_list)
    bic_dataframe.to_csv(r"C:\Users\xrw\Desktop\bic.csv")


    # print("avg_bic: ", total_bic_list)





# total_prob = np.load('prob_result/probability result2.npy')
# print(total_prob)


# get total
# prob_list = []
# for i in range(18):
#     p = show_p_and_c(i+1)
#     prob_list.append(p)
# plt.xticks(np.arange(1,19,1))
# plt.plot(np.arange(1,19,1), prob_list)
# plt.show()


# show_certain_p_and_c()
# show_certain_p_and_c2()

total_prob = []
for i in range(4, 8):
    prob_list = np.array(show_different_feature_certain_cluster_num(i)) * 100
    print("prob: ", prob_list)
    prob_list = np.round(prob_list, 3)
    total_prob.append(prob_list)
print(dataset_list.shape)
total_prob = np.array(total_prob).T
print('total_prob: ', total_prob.shape)
print(total_prob)


if __name__ == '__main__':
    all_element = np.array(
        ['zcr', 'total_energy', 'attack_energy', '5_part_energy', 'three_harmonic', '4_harmonic', '5_harmonic',
         'sp_harmonic', 'three_attack_harmonic', '4_attack_harmonic', '5_attack_harmonic', 'sp_attack_harmonic',
         '5part_three_harmonic', '5part_4_harmonic', '5part_5_harmonic', '5part_sp_harmonic'])
    # zcr 有效
    print("zcr: ")
    zcr_list = get_differ(['zcr'], 1)
    print(zcr_list)
    zcr = get_prob_diff(zcr_list)

    # 5th harmonic 无效
    print("5th harmonic: ")
    h5 = []
    hall5th_list = get_differ(['5_harmonic', '5_attack_harmonic', '5part_5_harmonic'], 3)
    print(hall5th_list)
    h5p_5th_list = get_differ(['5_harmonic', '5_attack_harmonic'], 2)
    print(h5p_5th_list)
    ha_5p_5th_List = get_differ(['5_harmonic'], 1)
    print(ha_5p_5th_List)
    h51 = get_prob_diff(hall5th_list)
    h52 = get_prob_diff(h5p_5th_list)
    h53 = get_prob_diff(ha_5p_5th_List)
    h5.extend(h51)
    h5.extend(h52)
    h5.extend(h53)
    h5 = np.array(h5)
    h5_mean = np.average(h5)

    # attack energy
    print("attack energy: ")
    attack_list = get_differ(['attack_energy'], 1)
    print(attack_list)
    a1 = get_prob_diff(attack_list)
    a1 = np.array(a1)
    a1_mean = np.average(a1)

    # sp
    sp = []
    print("sp: ")
    hall_sp_list = get_differ(['sp_harmonic', 'sp_attack_harmonic', '5part_sp_harmonic'], 3)
    print(hall_sp_list)
    h5p_sp_list = get_differ(['sp_harmonic', 'sp_attack_harmonic'], 2)
    print(h5p_sp_list)
    ha_5p_sp_list = get_differ(['sp_harmonic'], 1)
    print(ha_5p_sp_list)
    sp1 = get_prob_diff(hall_sp_list)
    sp2 = get_prob_diff(h5p_sp_list)
    sp3 = get_prob_diff(ha_5p_sp_list)
    sp.extend(sp1)
    sp.extend(sp2)
    sp.extend(sp3)

    # 4th
    print("4th harmonic: ")
    h4 = []
    hall4th_list = get_differ(['4_harmonic', '4_attack_harmonic', '5part_4_harmonic'], 3)
    print(hall4th_list)
    h5p_4th_list = get_differ(['4_harmonic', '4_attack_harmonic'], 2)
    print(h5p_4th_list)
    ha_5p_4th_List = get_differ(['4_harmonic'], 1)
    print(ha_5p_4th_List)
    h41 = get_prob_diff(hall4th_list)
    h42 = get_prob_diff(h5p_4th_list)
    h43 = get_prob_diff(ha_5p_4th_List)
    h4.extend(h41)
    h4.extend(h42)
    h4.extend(h43)
    h4 = np.array(h4)
    h4_mean = np.average(h4)

    total_list = [zcr, h4, h5, a1, sp]
    fig, ax = plt.subplots()
    ax.boxplot(total_list)
    ax.set_xticklabels(['zcr', '4th_harmonic', '5th_harmonic', 'attack_energy', "sp_harmonic"])
    plt.show()

    # # get bic and prob list
    # show_bic('')
    show_different_feature('')
    # total_prob = []
    # for i in range(4, 8):
    #     prob_list = np.array(show_different_feature_certain_cluster_num(i)) * 100
    #     print(len(prob_list))
    #     prob_list = np.round(prob_list, 2)
    #     total_prob.append(prob_list)
    # total_prob.append(dataset_list)
    # total_prob = np.array(total_prob).T
    # print('total_prob: ', total_prob.shape)
    # print(total_prob)
    # prob_df = pd.DataFrame(total_prob, columns=["4", "5", "6", "7", "dataset_name"])
    # prob_df.to_csv(r"C:\Users\xrw\Desktop\prob_list.csv")
    # plt.figure(figsize=(14, 10))
    # plt.title("4 clusters")
    # prob_df.sort_values(by=['4'], ascending=False)
    # plt.barh(prob_df["dataset_name"], prob_df['4'])
    # plt.show()


    # show_certain_p_and_c2(5)
    # show_p_and_c(18)


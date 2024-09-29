import os.path
import shutil

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



def plot_cluster_number_dis(best_component, instrument_list, best_cluster_labels, n_feature, cv_type='full'):
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

def mycopyfile(srcfile, dstpath):  # 复制函数
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        dst_file = os.path.join(dstpath, fname)
        shutil.copy(srcfile, dst_file)  # 复制文件
        print("copy %s -> %s" % (srcfile, dst_file))

def cluster_audio(csv_path, audio_path, out_put_path):
    dataset = pd.read_csv(csv_path)
    dataset = dataset.values
    cluster = dataset[:,-1]
    audio_path_list = os.listdir(audio_path)
    audio_path_list.sort(key=lambda x: int(x.split("-")[1]))
    print(audio_path_list)
    for i, c in enumerate(cluster):
        audio_note_path = os.path.join(audio_path, audio_path_list[i])
        cluster_path = os.path.join(out_put_path,"cluster_" + str(int(c)))
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        mycopyfile(audio_note_path, cluster_path)


if __name__ == '__main__':
    ab_path = r"C:\Users\xrw\Desktop\cluster_audio"
    csv_path = r"C:\Users\xrw\Desktop\4_14_10_6cluster.csv"
    audio_path = r"C:\Users\xrw\Desktop\note_audio"
    file_name = os.path.basename(csv_path).split('.')[0]
    out_put_path = os.path.join(ab_path, file_name)
    if not os.path.exists(out_put_path):
        os.makedirs(out_put_path)
    cluster_audio(csv_path, audio_path, out_put_path)
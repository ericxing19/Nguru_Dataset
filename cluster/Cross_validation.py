import pandas as pd
import clustering
import Standardization
import numpy as np

# csv_path = r"C:\Users\邢\Desktop\10 nguru recording\totaldataset_18_t.csv"
csv_path = r"C:\Users\xrw\Desktop\10 nguru recording\2023_2_16_median(48000hz_4096nfft)_processed.csv"


# def cross_validation(gmm, test_data):
#     row = test_data.shape[0]
#     column = test_data.shape[1]
#     r25 = int(row / 4)
#     r75 = int(3 * row / 4)
#     best_component = gmm.n_components
#     best_init = gmm.init_params
#     means = gmm.means_
#     label = gmm.predict(test_data)
#     correlation_list = []
#     eu_distance_list = []
#     for i in range(row):
#         u = means[label[i]]
#         u_mean = np.mean(u)
#         t = test_data[i]
#         t_mean = np.mean(t)
#         correlation = 0
#         for j in range(column):
#             correlation = correlation + (t[j] - t_mean)*(u[j] - u_mean)
#         correlation = correlation/(np.sqrt(np.sum(np.power(t-t_mean,2))) * np.sqrt(np.sum(np.power(u-u_mean,2))))
#         eu_distance = np.sqrt(np.sum(np.power(t - u, 2)))
#         correlation_list.append(correlation)
#         eu_distance_list.append(eu_distance)
#     sort_c_list = np.sort(correlation_list)
#     sort_eu_list = np.sort(eu_distance_list)
#     co_max = np.max(correlation_list)
#     co_min = np.min(correlation_list)
#     co_avg = np.average(correlation_list)
#     co_25 = sort_c_list[r25]
#     eu_max = np.max(eu_distance_list)#     co_75 = sort_c_list[r75]
#     eu_min = np.min(eu_distance_list)
#     eu_avg = np.average(eu_distance_list)
#     eu_25 = sort_eu_list[r25]
#     eu_75 = sort_eu_list[r75]
#     return sort_c_list,sort_eu_list,[co_avg,co_25,co_75,eu_avg,eu_25,eu_75]

def split_train_test(data):
    note_number = 151
    test_index = []
    for i in range(note_number):
        test_index.append(np.random.randint(5 * i, 5 * (i+1)))
    test_data = data[test_index]
    train_data = np.delete(data,test_index,axis=0)
    return np.array(train_data), np.array(test_data)

# aj = 1
def cal_probability(gmm, train_label, test_data):
    row = test_data.shape[0]
    column = test_data.shape[1]
    test_label = gmm.predict(test_data)


# directly multiply five probabilities
def cross_validation1(train_density, test_density):
    row = test_density.shape[0]
    # r25 = int(row / 4)
    # r75 = int(3 * row / 4)
    prob = np.zeros(151,)
    for i in range(151):
        one_note_train_prob = np.array(train_density[(4 * i) : (4*i+4)])
        one_note_test_prob = test_density[i]
        # print("train:", one_note_train_prob)
        # print("test: ", one_note_test_prob)
        probability = one_note_train_prob[0] * one_note_train_prob[1] * one_note_train_prob[2] * one_note_train_prob[3] * one_note_test_prob
        # prob[i] = np.sum(probability) , original method
        prob[i] = np.max(probability)
    # print("prob:", prob)
    avg_prob = np.average(prob)
    print("probability:", avg_prob)
    return avg_prob

# sum train and multiply with test
def cross_validation2(train_density, test_density):
    row = test_density.shape[0]
    # r25 = int(row / 4)
    # r75 = int(3 * row / 4)
    prob = np.zeros(151,)
    for i in range(151):
        one_note_train_prob = np.array(train_density[(4 * i) : (4*i+4)])
        one_note_test_prob = test_density[i]
        # print(one_note_train_prob)
        # print(one_note_test_prob)
        train_weight = np.sum(one_note_train_prob, axis = 0) /4
        probability = train_weight * one_note_test_prob
        prob[i] = np.sum(probability)
    # print("prob:", prob)
    avg_prob = np.average(prob)
    print("probability:", avg_prob)
    return avg_prob

# directly probability in one cluster
def cross_validation3(train_density, test_density):
    row = test_density.shape
    # r25 = int(row / 4)
    # r75 = int(3 * row / 4)
    prob = np.zeros(151, )
    for i in range(151):
        train_max_pos = np.argmax(train_density[(4 * i) : (4*i+4)], axis = 1)
        test_max_pos = np.argmax(test_density[i])
        max_pos = np.append(train_max_pos, test_max_pos)
        count_arr = np.bincount(max_pos)
        total_count = np.max(count_arr)
        prob[i] = total_count / 5
    avg_prob = np.average(prob)
    # print("probability list: ", prob)
    print("probability:", avg_prob)
    return  avg_prob

def traince1(use_data):
    ttimes = 50
    probability_list = np.zeros(50,)
    n_feature = use_data.shape[1]
    best_cluster = []
    for i in range(ttimes):
        train_data, test_data = split_train_test(use_data)
        bic, gmm = clustering.gmm10(use_data, train_data, test_data, if_plot, instument_list, 50, n_feature)
        best_cluster.append([gmm.n_components,bic])
        train_density = gmm.predict_proba(train_data)
        test_density = gmm.predict_proba(test_data)
        probability = cross_validation1(train_density, test_density)
        probability_list[i] = probability
    avg_probability = np.sum(probability_list)/ttimes
    return avg_probability, best_cluster, probability_list

def traince2(use_data):
    ttimes = 50
    probability_list = np.zeros(50, )
    n_feature = use_data.shape[1]
    best_cluster = []
    for i in range(ttimes):
        train_data, test_data = split_train_test(use_data)
        bic, gmm = clustering.gmm10(use_data, train_data, test_data, if_plot, instument_list, 20, n_feature)
        best_cluster.append([gmm.n_components, bic])
        train_density = gmm.predict_proba(train_data)
        test_density = gmm.predict_proba(test_data)
        probability = cross_validation2(train_density, test_density)
        probability_list[i] = probability
    avg_probability = np.sum(probability_list) / ttimes
    return avg_probability, best_cluster, probability_list

# need change
def train_certain_cluster(use_data, cluster_num):
    ttimes = 100
    probability_list = np.zeros(100, )
    n_feature = use_data.shape[1]
    bic_list = np.zeros(100,)
    for i in range(ttimes):
        train_data, test_data = split_train_test(use_data)
        bic, gmm = clustering.gmm_no_plot(use_data, train_data, test_data, cluster_num, 30)
        bic_list[i] = bic
        train_density = gmm.predict_proba(train_data)
        test_density = gmm.predict_proba(test_data)
        probability = cross_validation3(train_density, test_density)
        probability_list[i] = probability
    avg_probability = np.sum(probability_list) / ttimes
    avg_bic = np.sum(bic_list) / ttimes
    return avg_probability, probability_list, avg_bic

def plot1(use_data, dataset):
    probability, best_cluster, prob_list = traince1(use_data)
    labels = np.arange(0,n_feature+1, 1)
    dataset = dataset + 'score.csv'
    print(dataset + " probability: ", probability)
    return probability, best_cluster, prob_list

def plot2(use_data, name):
    probability, best_cluster, prob_list = traince2(use_data)
    labels = np.arange(0, n_feature + 1, 1)
    dataset = name + 'score.csv'
    print(dataset + " probability: ", probability)
    return probability, best_cluster, prob_list

def sum_prob(use_data, name):
    # cluster_num * 1
    probability_list = np.zeros(9,)
    # cluster_num * 50
    prob_distribution_list = np.zeros((9, 100))
    bic_list = np.zeros(9,)
    for i in np.arange(2,11):
        probability, prob_list, bic = train_certain_cluster(use_data, i)
        labels = np.arange(0, n_feature + 1, 1)
        probability_list[i-2] = probability
        prob_distribution_list[i-2] = prob_list
        bic_list[i - 2] = bic
        dataset = name + 'score.csv'
        print('cluster ' + str(i) + ', ' + dataset + " probability: " + str(probability) + ', ' + "avgbic: " + str(bic))
    return probability_list, prob_distribution_list, bic_list



def save_p(data, name):
    prob, c, p = plot2(data, name)
    print(name+': ', prob)
    print(name+': ', c)
    print(name+': ', p)
    np.save(name + ' c_num', c)
    np.save(name + ' p_list', p)
    np.save(name + ' prob', prob)

def save_certain_cluster_p(data, name):
    prob, prob_list, bic_list = sum_prob(data,name)
    print(name + ': ', prob)
    print(name + ': ', prob_list)
    np.save(name + ' certain_prob_list', prob)
    np.save(name + ' certain_prob_distribution_list', prob_list)
    np.save(name + ' bic_list', bic_list)

if __name__ == '__main__':
    dataset_list = ["LOG", "except_5th_harmonic", "except_4and5th_harmonic", "except_5parts_harmonic", "except_All5parts", "except_5th_5parts_harmonic", "except_4and5th_5parts_harmonic", "except_5th_All5parts", "except_4and5th_All5parts", "except_attack_5parts_harmonic", "except_attack_All5parts", "except_5th_attack_5parts_harmonic", "except_5th_attack_All5parts", "except_4and5th_attack_5parts_harmonic", "except_4and5th_attack_All5parts", "except_Allattack_All5parts", "except_5th_Allattack_All5parts", "except_4and5th_Allattack_All5parts", 'except_zcr', 'except_zcr_5th_harmonic', 'except_zcr_4and5th_harmonic', 'except_zcr_5parts_harmonic', 'except_sp', ' except_sp_5th_harmonic', 'except_sp_4and5th_harmonic', 'except_sp_5parts_harmonic', 'except_sp_attack_5parts_harmonic',  'except_sp_attack_All5parts', 'except_sp_4and5th_attack_All5parts', 'except_sp_4and5th_Allattack_All5parts']
    all_element = np.array(
        ['zcr', 'total_energy', 'attack_energy', '5_part_energy', 'three_harmonic', '4_harmonic', '5_harmonic',
         'sp_harmonic', 'three_attack_harmonic', '4_attack_harmonic', '5_attack_harmonic', 'sp_attack_harmonic',
         '5part_three_harmonic', '5part_4_harmonic', '5part_5_harmonic', '5part_sp_harmonic'])
    n_cluster = 16
    n_feature = 60
    feature_list = np.arange(n_feature+1)
    data = pd.read_csv(csv_path)
    dataset = data.values
    row_num = dataset.shape[0]
    # 1.four or one method  2. whether draw the picture 3. whether draw the average picture 4. whether write cluster number 5.whether show instrument list
    # One method
    if_plot = [False, False, False, False, False]
    # four method
    # if_plot = [True, True, False, False]
    # one method, average picture
    # if_plot = [False, False, True, False]
    if (dataset.shape[1] == n_feature + 1):
        data = dataset[:, :-1]
        instument_list = dataset[:, -1]
    elif (dataset.shape[1] == n_feature + 2):
        data = dataset[:, :-2]
        instument_list = dataset[:, -2]
    instument_list = np.squeeze(instument_list)
    MVR_data = Standardization.MVR_scale(data)
    MR_data = Standardization.MR_scale(data)
    log_data = Standardization.Log_scale(data)
    use_data = log_data


    # delete 4th and 5th harmonic
    except_5th_harmonic = np.delete(use_data, (11, 17, 22, 28, 50, 51, 52, 53, 54), axis=1)
    n_feature1 = 51
    except_4and5th_harmonic = np.delete(use_data, (10, 11, 16, 17, 21, 22, 27, 28, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54), axis=1)
    n_feature2 = 42
    # delete 1/5 parts
    except_5parts_harmonic = np.delete(use_data, np.arange(30, 60, 1), axis=1)
    n_feature3 = 30
    except_All5parts = np.delete(except_5parts_harmonic, (3,4,5,6,7), axis=1)
    n_feature4 = 25

    except_5th_5parts_harmonic = np.delete(except_5parts_harmonic, (11,17,22,28),axis=1)
    n_feature5 = 26
    except_4and5th_5parts_harmonic = np.delete(except_5parts_harmonic, (10,11,16,17,21,22,27,28), axis = 1)
    n_feature6 = 22

    except_5th_All5parts = np.delete(except_5th_5parts_harmonic, (3,4,5,6,7), axis = 1)
    n_feature7 = 21
    except_4and5th_All5parts = np.delete(except_4and5th_5parts_harmonic, (3,4,5,6,7), axis = 1)
    n_feature8 = 17

    # delete attack and 1/5 parts
    except_attack_5parts_harmonic = np.delete(except_5parts_harmonic, np.arange(19,30,1), axis = 1)
    n_feature9 = 19
    except_attack_All5parts = np.delete(except_attack_5parts_harmonic, (3,4,5,6,7), axis=1)
    n_feature10 = 14
    except_5th_attack_5parts_harmonic = np.delete(except_attack_5parts_harmonic, (11,17), axis=1)
    n_feature11 = 17
    except_5th_attack_All5parts = np.delete(except_attack_5parts_harmonic, (3,4,5,6,7,11,17), axis = 1)
    n_feature12 = 12
    element13 = np.delete(all_element, [3,6,8,9,10,11,12,13,14,15])
    except_4and5th_attack_5parts_harmonic = np.delete(except_attack_5parts_harmonic, (10,11,16,17), axis = 1)
    n_feature13 = 15
    element14 = np.delete(all_element, [5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
    except_4and5th_attack_All5parts = np.delete(except_attack_5parts_harmonic, (3,4,5,6,7,10,11,16,17), axis = 1)
    n_feature14 = 10
    element15 = np.delete(all_element, [3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])

    # delete all attack and all 1/5 parts
    except_Allattack_All5parts = np.delete(except_attack_All5parts, (2), axis = 1)
    n_feature15 = 13
    element16 = np.delete(all_element, [2, 3, 8, 9, 10, 11, 12, 13, 14, 15])
    except_5th_Allattack_All5parts = np.delete(except_5th_attack_All5parts, (2), axis = 1)
    n_feature16 = 11
    element17 = np.delete(all_element, [2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15])
    except_4and5th_Allattack_All5parts = np.delete(except_4and5th_attack_All5parts, (2), axis = 1)
    n_feature17 = 9
    element18 = np.delete(all_element, [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
    except_absolute = np.delete(use_data, np.arange(0, 8, 1), axis = 1)
    n_feature18 = 52
    element19 = np.delete(all_element, [0, 1, 2, 3])
    except_absolute_5th = np.delete(except_5th_harmonic,np.arange(0,8,1), axis = 1)
    n_feature19 = 43
    element20 = np.delete(all_element, [0,1,2,3,6,10,14])
    except_absolute_4and5th = np.delete(except_4and5th_harmonic, np.arange(0, 8, 1), axis = 1)
    n_feature20 = 34
    element21 = np.delete(all_element, [0, 1, 2, 3, 5, 6, 9, 10, 13, 14])
    except_absolute_5parts_harmonic = np.delete(except_5parts_harmonic, np.arange(0, 8, 1), axis=1)
    n_feature21 = 22
    element22 = np.delete(all_element, [0, 1, 2, 3, 12, 13, 14, 15])
    except_absolute_5th_5parts_harmonic = np.delete(except_5th_5parts_harmonic, np.arange(0, 8, 1), axis=1)
    n_feature22 = 18
    element23 = np.delete(all_element, [0, 1, 2, 3, 6, 10, 12, 13, 14, 15])
    except_absolute_4and5th_5parts_harmonic = np.delete(except_4and5th_5parts_harmonic, np.arange(0, 8, 1), axis=1)
    n_feature23 = 14
    element24 = np.delete(all_element, [0, 1, 2, 3, 5, 6, 9, 10, 12, 13, 14, 15])
    except_absolute_attack_5parts_harmonic = np.delete(except_attack_5parts_harmonic, np.arange(0, 8, 1), axis= 1)
    n_feature24 = 11
    element25 = np.delete(all_element, [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15])
    except_absolute_5th_attack_5parts_harmonic = np.delete(except_5th_attack_5parts_harmonic, np.arange(0, 8, 1), axis= 1)
    n_feature25 = 9
    element26 = np.delete(all_element, [0, 1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15])
    except_absolute_4and5th_attack_5parts_harmonic = np.delete(except_4and5th_attack_5parts_harmonic, np.arange(0, 8, 1), axis= 1)
    n_feature26 = 7
    element27 = np.delete(all_element, [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15])
    # except zcr
    except_zcr = np.delete(use_data, (0), axis = 1)
    n_feature27 = 59
    element28 = np.delete(all_element, [0])
    except_zcr_5th_harmonic = np.delete(use_data, (0, 11, 17, 22, 28, 50, 51, 52, 53, 54), axis = 1)
    n_feature28 = 50
    element29 = np.delete(all_element, [0, 6, 10, 14])
    except_zcr_4and5th_harmonic = np.delete(use_data, (0, 10, 11, 16, 17, 21, 22, 27, 28, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54), axis = 1)
    n_feature29 = 41
    element30 = np.delete(all_element, [0, 5, 6, 9, 10, 13, 14])
    except_zcr_5parts_harmonic = np.delete(except_5parts_harmonic, (0), axis = 1)
    n_feature30 = 29
    element31 = np.delete(all_element, [0, 12, 13, 14, 15])
    # except sp
    except_sp = np.delete(use_data, (12, 18, 23, 29, 55, 56, 57, 58, 59), axis=1)
    n_feature31 = 51
    element32 = np.delete(all_element, [7,11,15])
    except_sp_5th_harmonic = np.delete(use_data, (11,12,17,18,22,23,28,29,50,51,52,53,54,55,56,57,58,59), axis = 1)
    n_feature32 = 42
    element33 = np.delete(all_element, [6, 7, 10, 11, 14, 15])
    except_sp_4and5th_harmonic = np.delete(use_data, (10, 11, 12, 16, 17, 18, 21, 22, 23, 27, 28, 29, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,55,56,57,58,59), axis = 1)
    n_feature33 = 33
    element34 = np.delete(all_element, [5, 6, 7, 9, 10, 11, 13, 14, 15])
    except_sp_5parts_harmonic = np.delete(except_5parts_harmonic, (12,18,23,29), axis = 1)
    n_feature34 = 26
    element35 = np.delete(all_element, [7, 11, 12, 13, 14, 15])
    except_sp_attack_5parts_harmonic = np.delete(except_attack_5parts_harmonic, (12, 18), axis = 1)
    n_feature35 = 17
    element36 = np.delete(all_element, [7, 8, 9, 10, 11, 12, 13, 14, 15])
    except_sp_attack_All5parts = np.delete(except_sp_attack_5parts_harmonic, (3,4,5,6,7), axis = 1)
    n_feature36 = 12
    element37 = np.delete(all_element, [3, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    except_sp_4and5th_attack_All5parts = np.delete(except_attack_5parts_harmonic, (3,4,5,6,7,10,11,12,16,17,18), axis = 1)
    n_feature37 = 8
    element38 = np.delete(all_element, [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    except_sp_4and5th_Allattack_All5parts = np.delete(except_sp_4and5th_attack_All5parts, (2), axis = 1)
    n_feature38 = 7
    element39 = np.delete(all_element, [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


    # prob1, c1b,p1l = plot1(use_data,"MR")
    # prob2, c2b,p2l = plot1(except_5th_harmonic,"except_5th_harmonic")
    # prob3, c3b,p3l = plot1(except_4and5th_harmonic, "except_4and5th_harmonic")
    # prob4, c4b,p4l = plot1(except_5parts_harmonic, "except_5parts_harmonic")
    # prob5, c5b, p5l = plot1(except_All5parts, "except_All5parts")
    # prob6, c6b, p6l = plot1(except_5th_5parts_harmonic, "except_5th_5parts_harmonic")
    # prob7, c7b, p7l = plot1(except_4and5th_5parts_harmonic, "except_4and5th_5parts_harmonic")
    # prob8, c8b, p8l = plot1(except_5th_All5parts, "except_5th_All5parts")
    # prob9, c9b, p9l = plot1(except_4and5th_All5parts, "except_4and5th_All5parts")
    # prob10, c10b, p10l = plot1(except_attack_5parts_harmonic, "except_attack_5parts_harmonic")
    # prob11, c11b, p11l = plot1(except_attack_All5parts, "except_attack_All5parts")
    # prob12, c12b, p12l = plot1(except_5th_attack_5parts_harmonic, "except_5th_attack_5parts_harmonic")
    # prob13, c13b, p13l = plot1(except_5th_attack_All5parts, "except_5th_attack_All5parts")
    # prob14, c14b, p14l = plot1(except_4and5th_attack_5parts_harmonic, "except_4and5th_attack_5parts_harmonic")
    # prob15, c15b, p15l = plot1(except_4and5th_attack_All5parts, "except_4and5th_attack_All5parts")
    # prob16, c16b, p16l = plot1(except_Allattack_All5parts, "except_Allattack_All5parts")
    # prob17, c17b, p17l = plot1(except_5th_Allattack_All5parts, "except_5th_Allattack_All5parts")
    # prob18, c18b, p18l = plot1(except_4and5th_Allattack_All5parts, "except_4and5th_Allattack_All5parts")
    #
    #
    #
    #

    # total_prob = np.array([prob1,prob2,prob3,prob4,prob5,prob6,prob7,prob8,prob9,prob10,prob11,prob12,prob13,prob14,prob15,prob16,prob17,prob18])
    #
    # print("total prob",total_prob)
    # np.save("probability result2", arr = total_prob)
    # np.save("c1b", arr = c1b)
    # np.save("c2b", arr=c2b)
    # np.save("c3b", arr=c3b)
    # np.save("c4b", arr=c4b)
    # np.save("c5b", arr=c5b)
    # np.save("c6b", arr=c6b)
    # np.save("c7b", arr=c7b)
    # np.save("c8b", arr=c8b)
    # np.save("c9b", arr=c9b)
    # np.save("c10b", arr=c10b)
    # np.save("c11b", arr=c11b)
    # np.save("c12b", arr=c12b)
    # np.save("c13b", arr=c13b)
    # np.save("c14b", arr=c14b)
    # np.save("c15b", arr=c15b)
    # np.save("c16b", arr=c16b)
    # np.save("c17b", arr=c17b)
    # np.save("c18b", arr=c18b)
    #
    # np.save("p1b", arr=p1l)
    # np.save("p2b", arr=p2l)
    # np.save("p3b", arr=p3l)
    # np.save("p4b", arr=p4l)
    # np.save("p5b", arr=p5l)
    # np.save("p6b", arr=p6l)
    # np.save("p7b", arr=p7l)
    # np.save("p8b", arr=p8l)
    # np.save("p9b", arr=p9l)
    # np.save("p10b", arr=p10l)
    # np.save("p11b", arr=p11l)
    # np.save("p12b", arr=p12l)
    # np.save("p13b", arr=p13l)
    # np.save("p14b", arr=p14l)
    # np.save("p15b", arr=p15l)
    # np.save("p16b", arr=p16l)
    # np.save("p17b", arr=p17l)
    # np.save("p18b", arr=p18l)


    # save_p(use_data, "MR")
    # save_p(except_5th_harmonic, "except_5th_harmonic")
    # save_p(except_4and5th_harmonic, "except_4and5th_harmonic")
    # save_p(except_5parts_harmonic, "except_5parts_harmonic")
    # save_p(except_All5parts, "except_All5parts")
    # save_p(except_5th_5parts_harmonic, "except_5th_5parts_harmonic")
    # save_p(except_4and5th_5parts_harmonic, "except_4and5th_5parts_harmonic")
    # save_p(except_5th_All5parts, "except_5th_All5parts")
    # save_p(except_4and5th_All5parts, "except_4and5th_All5parts")
    # save_p(except_attack_5parts_harmonic, "except_attack_5parts_harmonic")
    # save_p(except_attack_All5parts, "except_attack_All5parts")
    # save_p(except_5th_attack_5parts_harmonic, "except_5th_attack_5parts_harmonic")
    # save_p(except_5th_attack_All5parts, "except_5th_attack_All5parts")
    # save_p(except_4and5th_attack_5parts_harmonic, "except_4and5th_attack_5parts_harmonic")
    # save_p(except_4and5th_attack_All5parts, "except_4and5th_attack_All5parts")
    # save_p(except_Allattack_All5parts, "except_Allattack_All5parts")
    # save_p(except_5th_Allattack_All5parts, "except_5th_Allattack_All5parts")
    # save_p(except_4and5th_Allattack_All5parts, "except_4and5th_Allattack_All5parts")


    # save_certain_cluster_p(use_data, "LOG")
    # save_certain_cluster_p(except_5th_harmonic, "except_5th_harmonic")
    # save_certain_cluster_p(except_4and5th_harmonic, "except_4and5th_harmonic")
    # save_certain_cluster_p(except_5parts_harmonic, "except_5parts_harmonic")
    # save_certain_cluster_p(except_All5parts, "except_All5parts")
    # save_certain_cluster_p(except_5th_5parts_harmonic, "except_5th_5parts_harmonic")
    # save_certain_cluster_p(except_4and5th_5parts_harmonic, "except_4and5th_5parts_harmonic")
    # save_certain_cluster_p(except_5th_All5parts, "except_5th_All5parts")
    # save_certain_cluster_p(except_4and5th_All5parts, "except_4and5th_All5parts")
    # save_certain_cluster_p(except_attack_5parts_harmonic, "except_attack_5parts_harmonic")
    # save_certain_cluster_p(except_attack_All5parts, "except_attack_All5parts")
    # save_certain_cluster_p(except_5th_attack_5parts_harmonic, "except_5th_attack_5parts_harmonic")
    # save_certain_cluster_p(except_5th_attack_All5parts, "except_5th_attack_All5parts")
    # save_certain_cluster_p(except_4and5th_attack_5parts_harmonic, "except_4and5th_attack_5parts_harmonic")
    # save_certain_cluster_p(except_4and5th_attack_All5parts, "except_4and5th_attack_All5parts")
    # save_certain_cluster_p(except_Allattack_All5parts, "except_Allattack_All5parts")
    # save_certain_cluster_p(except_5th_Allattack_All5parts, "except_5th_Allattack_All5parts")
    # save_certain_cluster_p(except_4and5th_Allattack_All5parts, "except_4and5th_Allattack_All5parts")
    # save_certain_cluster_p(except_absolute, "except_absolute")
    # save_certain_cluster_p(except_absolute_5th, "except_absolute_5th")
    # save_certain_cluster_p(except_absolute_4and5th, "except_absolute_4and5th")
    # save_certain_cluster_p(except_absolute_5parts_harmonic, "except_absolute_5parts_harmonic")
    # save_certain_cluster_p(except_absolute_5th_5parts_harmonic, "except_absolute_5th_5parts_harmonic")
    # save_certain_cluster_p(except_absolute_4and5th_5parts_harmonic, "except_absolute_4and5th_5parts_harmonic")
    # save_certain_cluster_p(except_absolute_attack_5parts_harmonic, "except_absolute_attack_5parts_harmonic")
    # save_certain_cluster_p(except_absolute_5th_attack_5parts_harmonic, "except_absolute_5th_attack_5parts_harmonic")
    # save_certain_cluster_p(except_absolute_4and5th_attack_5parts_harmonic, "except_absolute_4and5th_attack_5parts_harmonic")
    # save_certain_cluster_p(except_zcr, "except_zcr")
    # save_certain_cluster_p(except_zcr_5th_harmonic, 'except_zcr_5th_harmonic')
    # save_certain_cluster_p(except_zcr_4and5th_harmonic, 'except_zcr_4and5th_harmonic')
    # save_certain_cluster_p(except_zcr_5parts_harmonic, 'except_zcr_5parts_harmonic')
    save_certain_cluster_p(except_sp, 'except_sp')
    save_certain_cluster_p(except_sp_5th_harmonic, 'except_sp_5th_harmonic')
    save_certain_cluster_p(except_sp_4and5th_harmonic, 'except_sp_4and5th_harmonic')
    save_certain_cluster_p(except_sp_5parts_harmonic, 'except_sp_5parts_harmonic')
    save_certain_cluster_p(except_sp_attack_5parts_harmonic, 'except_sp_attack_5parts_harmonic')
    save_certain_cluster_p(except_sp_attack_All5parts, 'except_sp_attack_All5parts')
    save_certain_cluster_p(except_sp_4and5th_attack_All5parts, 'except_sp_4and5th_attack_All5parts')
    save_certain_cluster_p(except_sp_4and5th_Allattack_All5parts, 'except_sp_4and5th_Allattack_All5parts')


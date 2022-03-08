import os
import pandas as pd
from statistics import mean
from scipy.stats import pearsonr
from numpy import average
import numpy as np
from collections import Counter


def get_data(my_dir):
    annotations_df = pd.read_csv(my_dir + '/annotations.csv')
    features_df = pd.read_csv(my_dir + '/mean_df.csv')
    return annotations_df, features_df


def get_users_dic(annotations_df):
    users_dic_A = {}
    users_dic_V = {}
    GS_dic = {}
    list_ID = annotations_df["ID"].unique()
    for ID in list_ID:
        user_df = annotations_df[annotations_df["ID"]==ID]
        users_dic_A[ID] = user_df["Arousal"].tolist()
        users_dic_V[ID] = user_df["Valence"].tolist()
        songs_list = user_df["Song2"].tolist()

    GS_dic["Song2"] = songs_list
    # put the data in a list of lists
    A_vals = [seq for seq in users_dic_A.values()]
    V_vals = [seq for seq in users_dic_V.values()]
    mean_list_A = [mean(i) for i in zip(*A_vals)]
    mean_list_V = [mean(i) for i in zip(*V_vals)]
    GS_dic["A_mean"] = [round(num, 2) for num in mean_list_A]
    GS_dic["V_mean"] = [round(num, 2) for num in mean_list_V]
    return users_dic_A, users_dic_V, GS_dic


def get_weights(users_dic_A, users_dic_V, GS_dic):
    users_weights_A = {}
    users_weights_V = {}
    for key in users_dic_A:
        correlation, p_value = pearsonr(users_dic_A[key], GS_dic["A_mean"])
        users_weights_A[key] = abs(correlation)
        correlation, p_value = pearsonr(users_dic_V[key], GS_dic["V_mean"])
        users_weights_V[key] = abs(correlation)
    return users_weights_A, users_weights_V


def get_EWE_vals_per_song(users_dic, index, users_weights, EWE_list):
    list_ratings = []
    list_weights = []
    for key in users_dic:
        list_ratings.append(users_dic[key][index])
        list_weights.append(users_weights[key])
    EWE_list.append(round(average(list_ratings, weights=list_weights), 2))
    return EWE_list


def get_EWE(users_dic_A, users_dic_V, GS_dic, users_weights_A, users_weights_V):
    EWE_list_A = []
    EWE_list_V = []
    for index, song in enumerate(GS_dic["Song2"]):
        EWE_list_A = get_EWE_vals_per_song(users_dic_A, index, users_weights_A, EWE_list_A)
        EWE_list_V = get_EWE_vals_per_song(users_dic_V, index, users_weights_V, EWE_list_V)
    GS_dic["EWE_A"] = EWE_list_A
    GS_dic["EWE_V"] = EWE_list_V
    return GS_dic


def get_category(GS_dic, annotations_df):
    for song in GS_dic['Song2']:
        song_df = annotations_df[annotations_df["Song2"]==song]
        assigned_emotions = song_df["Category"].tolist()
        c = Counter(assigned_emotions)
        if 'Category' not in GS_dic:
            GS_dic['Category'] = []
            GS_dic['Category'].append(c.most_common(1)[0][0])
        else:
            GS_dic['Category'].append(c.most_common(1)[0][0])
    return GS_dic


def rename_songs(GS_dic):
    list_songs1 = GS_dic['Song2']
    list_songs2 = []
    dic_names = {"Bor": "bor_ps1", "Alb": "alb_esp1", "Muss": "muss_1", "Bal": "bal_islamei", "Scn_15": "scn_15_6",
                 "Br_im6": "br_im6", "Br_im2": "br_im2", "Bach_847": "bach_847", "Schub_143": "Schub_143_1",
                 "Beet_elise": "beet_elise", "Gr": "gr_kobold", "Beet_pathetique": "beet_pathetique_1", "Deb": "deb_pass",
                 "Ty_januar": "ty_januar", "Chpn_op10": "chpn_op10_e01", "Mz_333": "mz_333_1", "Mz_330": "mz_330_1",
                 "Bach_850": "bach_850", "Schub_d760": "schub_d760_4", "Chpn_op7": "chpn_op7_1", "Liz": "liz_et1",
                 "Ty_april": "ty_april", "Gra": "gra_esp_2", "Scn_4": "scn_4"}
    for elem in list_songs1:
        if elem in dic_names:
            new_name = dic_names[elem]
            list_songs2.append(new_name)
    GS_dic['Song'] = list_songs2
    return GS_dic


def get_Q(GS_dic, string_A, string_V, string_Q):
    Q = []
    mean_A = np.mean([min(GS_dic[string_A]), max(GS_dic[string_A])])
    mean_V = np.mean([min(GS_dic[string_V]), max(GS_dic[string_V])])
    for index, rating in enumerate(GS_dic[string_A]):
        if rating > mean_A and GS_dic[string_V][index] > mean_V:
            Q.append('Q1')
        elif rating > mean_A and GS_dic[string_V][index] < mean_V:
            Q.append('Q2')
        elif rating < mean_A and GS_dic[string_V][index] < mean_V:
            Q.append('Q3')
        else:
            Q.append('Q4')
    GS_dic[string_Q] = Q
    return GS_dic


if __name__ == "__main__":
    my_dir = os.getcwd()
    annotations_df, features_df = get_data(my_dir)
    users_dic_A, users_dic_V, GS_dic = get_users_dic(annotations_df)
    users_weights_A, users_weights_V = get_weights(users_dic_A, users_dic_V, GS_dic)
    GS_dic = get_EWE(users_dic_A, users_dic_V, GS_dic, users_weights_A, users_weights_V)
    GS_dic = get_category(GS_dic, annotations_df)
    GS_dic = rename_songs(GS_dic)
    GS_dic = get_Q(GS_dic, "EWE_A", "EWE_V", "Q_EWE")
    GS_dic = get_Q(GS_dic, "A_mean", "V_mean", "Q_mean")

    df_labels = pd.DataFrame.from_dict(GS_dic)
    print(df_labels)
    features_df = features_df.drop(['Unnamed: 0', 'Arousal', 'Valence'], axis=1)
    print(features_df)
    Gold_Standard = pd.merge(df_labels, features_df, on='Song', how='inner')
    first_column = Gold_Standard.pop('Song')
    Gold_Standard.insert(0, 'Song', first_column)
    print(Gold_Standard)

    Gold_Standard.to_csv(my_dir + '/GoldStandard.csv')

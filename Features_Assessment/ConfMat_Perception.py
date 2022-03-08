import os
import pandas as pd
import numpy as np


def get_data(my_dir):
    annotations_df = pd.read_csv(my_dir + '/annotations.csv')
    gold_standard_df = pd.read_csv(my_dir + '/GoldStandard.csv')
    return annotations_df, gold_standard_df


def get_counts(results_dys, results_joy, results_tra, results_sad, results_act, annotations_df, gold_standard_df, GS_songs, GS_target):
    for index, song in enumerate(GS_songs[:-1]):
        target = GS_target[index]
        df_song = annotations_df[annotations_df.Song2 == song]
        counts = df_song.Category.value_counts()
        if target == 'Dysphoria':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_dys:
                    results_dys[counts.index[x]] = results_dys[counts.index[x]] + counts[x]
                else:
                    results_dys[counts.index[x]] = counts[x]
        elif target == 'Joy':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_joy:
                    results_joy[counts.index[x]] = results_joy[counts.index[x]] + counts[x]
                else:
                    results_joy[counts.index[x]] = counts[x]
        elif target == 'Tranquility':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_tra:
                    results_tra[counts.index[x]] = results_tra[counts.index[x]] + counts[x]
                else:
                    results_tra[counts.index[x]] = counts[x]
        elif target == 'Sadness':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_sad:
                    results_sad[counts.index[x]] = results_sad[counts.index[x]] + counts[x]
                else:
                    results_sad[counts.index[x]] = counts[x]
        elif target == 'Activation':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_act:
                    results_act[counts.index[x]] = results_act[counts.index[x]] + counts[x]
                else:
                    results_act[counts.index[x]] = counts[x]
    return results_dys, results_joy, results_tra, results_sad, results_act


if __name__ == "__main__":
    my_dir = os.getcwd()
    annotations_df, gold_standard_df = get_data(my_dir)
    GS_songs = gold_standard_df["Song2"].to_list()
    GS_target = gold_standard_df["Category"].to_list()
    results_dys = {}
    results_joy = {}
    results_tra = {}
    results_sad = {}
    results_act = {}
    results_dys, results_joy, results_tra, results_sad, results_act = get_counts(results_dys, results_joy, results_tra, results_sad, results_act, annotations_df, gold_standard_df, GS_songs, GS_target)
    print(results_joy)
    print(results_dys)
    print(results_sad)
    print(results_tra)
    print(results_act)
    CM = np.array([[results_joy['Joy'], results_joy['Dysphoria'], results_joy['Sadness'], results_joy['Tranquility'], results_joy['Activation'], results_joy['Power'], results_joy['TenderLonging'], results_joy['Amazement'], 0, results_joy['Sensuality']],
                   [0, results_dys['Dysphoria'], results_dys['Sadness'], results_dys['Tranquility'], results_dys['Activation'], results_dys['Power'], results_dys['TenderLonging'], results_dys['Amazement'], results_dys['Transcendence'], results_dys['Sensuality']],
                   [results_sad['Joy'], results_sad['Dysphoria'], results_sad['Sadness'], results_sad['Tranquility'], 0, 0, results_sad['TenderLonging'], results_sad['Amazement'], results_sad['Transcendence'], results_sad['Sensuality']],
                   [results_tra['Joy'], results_tra['Dysphoria'], results_tra['Sadness'], results_tra['Tranquility'], results_tra['Activation'], results_tra['Power'], results_tra['TenderLonging'], results_tra['Amazement'], results_tra['Transcendence'], results_tra['Sensuality']],
                   [results_act['Joy'], results_act['Dysphoria'], results_act['Sadness'], results_act['Tranquility'], results_act['Activation'], results_act['Power'], results_act['TenderLonging'], results_act['Amazement'], results_act['Transcendence'], results_act['Sensuality']]])

    print("Joy, Dysphoria, Sadness, Tranquility, Activation, Power, TenderLonging, Amazement, Transcendence, Sensuality")
    print(CM)
    new_CM = np.true_divide(CM, CM.sum(axis=1, keepdims=True))
    new_CM = new_CM*100
    print(np.around(new_CM, decimals=1, out=None))

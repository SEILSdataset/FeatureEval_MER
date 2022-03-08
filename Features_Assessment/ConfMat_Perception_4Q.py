import os
import pandas as pd
import numpy as np


def get_data(my_dir):
    annotations_df = pd.read_csv(my_dir + '/annotations.csv')
    gold_standard_df = pd.read_csv(my_dir + '/GoldStandard.csv')
    return annotations_df, gold_standard_df


def get_perception_Q(annotations_df):
    Arousal = annotations_df["Arousal"].to_list()
    Valence = annotations_df["Valence"].to_list()
    Q = []
    mean_A = np.mean([min(Arousal), max(Arousal)])
    mean_V = np.mean([min(Valence), max(Valence)])
    for index, rating in enumerate(Arousal):
        if rating > mean_A and Valence[index] > mean_V:
            Q.append('Q1')
        elif rating > mean_A and Valence[index] < mean_V:
            Q.append('Q2')
        elif rating < mean_A and Valence[index] < mean_V:
            Q.append('Q3')
        else:
            Q.append('Q4')
    return Q


def get_counts(results_Q1, results_Q2, results_Q3, results_Q4, annotations_df, gold_standard_df, GS_songs, GS_target):
    for index, song in enumerate(GS_songs[:-1]):
        target = GS_target[index]
        df_song = annotations_df[annotations_df.Song2 == song]
        counts = df_song.Q.value_counts()
        if target == 'Q1':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_Q1:
                    results_Q1[counts.index[x]] = results_Q1[counts.index[x]] + counts[x]
                else:
                    results_Q1[counts.index[x]] = counts[x]
        elif target == 'Q2':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_Q2:
                    results_Q2[counts.index[x]] = results_Q2[counts.index[x]] + counts[x]
                else:
                    results_Q2[counts.index[x]] = counts[x]
        elif target == 'Q3':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_Q3:
                    results_Q3[counts.index[x]] = results_Q3[counts.index[x]] + counts[x]
                else:
                    results_Q3[counts.index[x]] = counts[x]
        elif target == 'Q4':
            for x in range(0, len(counts.index)):
                if counts.index[x] in results_Q4:
                    results_Q4[counts.index[x]] = results_Q4[counts.index[x]] + counts[x]
                else:
                    results_Q4[counts.index[x]] = counts[x]
    return results_Q1, results_Q2, results_Q3, results_Q4


if __name__ == "__main__":
    my_dir = os.getcwd()
    annotations_df, gold_standard_df = get_data(my_dir)
    GS_songs = gold_standard_df["Song2"].to_list()
    GS_target = gold_standard_df["Q_EWE"].to_list()
    results_Q1 = {}
    results_Q2 = {}
    results_Q3 = {}
    results_Q4 = {}
    Q = get_perception_Q(annotations_df)
    annotations_df['Q'] = Q
    results_Q1, results_Q2, results_Q3, results_Q4 = get_counts(results_Q1, results_Q2, results_Q3, results_Q4, annotations_df, gold_standard_df, GS_songs, GS_target)
    print(results_Q1)
    print(results_Q2)
    print(results_Q3)
    print(results_Q4)
    CM = np.array([[results_Q1['Q1'], results_Q1['Q2'], results_Q1['Q3'], results_Q1['Q4']],
                   [results_Q2['Q1'], results_Q2['Q2'], results_Q2['Q3'], results_Q2['Q4']],
                   [results_Q3['Q1'], results_Q3['Q2'], results_Q3['Q3'], results_Q3['Q4']],
                   [results_Q4['Q1'], results_Q4['Q2'], results_Q4['Q3'], results_Q4['Q4']]])

    print("Q1, Q2, Q3, Q4")
    print(CM)
    new_CM = np.true_divide(CM, CM.sum(axis=1, keepdims=True))
    new_CM = new_CM*100
    print(np.around(new_CM, decimals=1, out=None))

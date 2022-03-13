import numpy as np
import pandas as pd
import scipy.stats
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import neural_network
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn import svm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def get_data(my_dir):
    df = pd.read_csv(my_dir + '/GoldStandard.csv')
    A_mean = df.iloc[:-1, 3:4].values  # Drop last row as contains song with unique label
    V_mean = df.iloc[:-1, 4:5].values
    A_EWE = df.iloc[:-1, 5:6].values
    V_EWE = df.iloc[:-1, 6:7].values
    y_col = df.iloc[:-1, 7:8]
    y_col = y_col.replace('Joy', 0)
    y_col = y_col.replace('Dysphoria', 1)
    y_col = y_col.replace('Sadness', 2)
    y_col = y_col.replace('Tranquility', 3)
    y_col = y_col.replace('Activation', 4)
    y = y_col.values
    x = df.iloc[:-1, 10:]
    A_selection_Cor = ["Most_Common_Rhythmic_Value", "Number_of_Strong_Rhythmic_Pulses_._Tempo_Standardized", "BPM", "pcm_zcr_sma_skewness", "Note_Density", "Standard_Triads", "Melodic_Large_Intervals", "Most_Common_Vertical_Interval", "Strength_Ratio_of_Two_Strongest_Rhythmic_Pulses_._Tempo_Standardized", "pcm_intensity_sma_min", "Prevalence_of_Dotted_Notes"]
    V_selection_Cor = ["F0env_sma_quartile3", "Minor_Major_Triad_Ratio", "Minor_Major_Melodic_Third_Ratio", "F0env_sma_skewness", "pcm_intensity_sma_minPos", "Similar_Motion", "Strength_Ratio_of_Two_Strongest_Rhythmic_Pulses_._Tempo_Standardized", "Diminished_and_Augmented_Triads", "Dynamic_Range", "Non.Standard_Chords", "pcm_zcr_sma_skewness"]
    A_selection_LR = ["Most_Common_Rhythmic_Value", "Number_of_Strong_Rhythmic_Pulses_._Tempo_Standardized", "Median_Rhythmic_Value_Offset", "pcm_zcr_sma_skewness", "pcm_zcr_sma_stddev", "F0env_sma_linregerrQ", "Non.Standard_Chords"]
    V_selection_LR = ["Minor_Major_Triad_Ratio", "pcm_zcr_sma_skewness", "F0env_sma_skewness", "F0env_sma_linregerrQ", "Amount_of_Staccato", "Strength_Ratio_of_Two_Strongest_Rhythmic_Pulses_._Tempo_Standardized", "BPM"]

    A_selection_Union = A_selection_Cor.copy()
    for elem in A_selection_LR:
        if elem not in A_selection_Union:
            A_selection_Union.append(elem)
    V_selection_Union = V_selection_Cor.copy()
    for elem in V_selection_LR:
        if elem not in V_selection_Union:
            V_selection_Union.append(elem)

    A_selection = A_selection_Union
    V_selection = V_selection_Union

    A_selection_index = [x.columns.get_loc(c) for c in A_selection if c in x]  # get indexes of selected features
    V_selection_index = [x.columns.get_loc(c) for c in V_selection if c in x]
    all_selection_index = A_selection_index.copy()
    for elem in V_selection_index:
        if elem not in all_selection_index:
            all_selection_index.append(elem)
    return df, A_mean, V_mean, A_EWE, V_EWE, y, x, A_selection_index, V_selection_index, all_selection_index


def some_descriptive(x, y):
    print("---------------------------")
    print('DATA SET')
    print(x.shape[0], 'instances from', len(np.unique(y)), 'classes', np.unique(y))
    print(x.shape[1], 'features')


def partitioning(x, y, A_selection_index, V_selection_index, all_selection_index, rand):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rand, stratify=y)
    scaler = StandardScaler()  # normalising features in the training set
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # Applies the same scaling and shifting operations performed on the train data
    y_train = y_train.flatten()  # get correct dimension (1D array)
    y_test = y_test.flatten()
    if rand == 0:
        print('TRAINING SET')
        print(x_train.shape[0], 'instances from', len(np.unique(y_train)), 'classes:', np.unique(y_train))
        print(Counter(y_train.tolist()))
        print(x_train.shape[1], ' features')
        print('TEST SET')
        print(x_test.shape[0], 'instances from', len(np.unique(y_test)), 'classes:', np.unique(y_test))
        print(Counter(y_test.tolist()))
        print(x_test.shape[1], ' features')
    SELarousal_test = np.take(x_test, A_selection_index, 1)  # filter out array according to the indexes of selected features
    SELarousal_train = np.take(x_train, A_selection_index, 1)
    SELvalence_test = np.take(x_test, V_selection_index, 1)
    SELvalence_train = np.take(x_train, V_selection_index, 1)
    SELall_test = np.take(x_test, all_selection_index, 1)
    SELall_train = np.take(x_train, all_selection_index, 1)
    return x_train, x_test, y_train, y_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, SELall_test, SELall_train


def SVM(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall):
    pred = np.zeros((len(y_test),4))
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(x_train, y_train).predict(x_test)
    pred[:,0] = predictions
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(SELarousal_train, y_train).predict(SELarousal_test)
    pred[:,1] = predictions
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(SELvalence_train, y_train).predict(SELvalence_test)
    pred[:,2] = predictions
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(SELall_train, y_train).predict(SELall_test)
    pred[:,3] = predictions
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred


def knn(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall):
    pred = np.zeros((len(y_test),4))
    predictions = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train).predict(x_test)
    pred[:,0] = predictions
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = KNeighborsClassifier(n_neighbors=5).fit(SELarousal_train, y_train).predict(SELarousal_test)
    pred[:,1] = predictions
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = KNeighborsClassifier(n_neighbors=5).fit(SELvalence_train, y_train).predict(SELvalence_test)
    pred[:,2] = predictions
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = KNeighborsClassifier(n_neighbors=5).fit(SELall_train, y_train).predict(SELall_test)
    pred[:,3] = predictions
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred


def RF(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall):
    pred = np.zeros((len(y_test),4))
    predictions = RandomForestClassifier(random_state=0).fit(x_train, y_train).predict(x_test)
    pred[:,0] = predictions
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = RandomForestClassifier(random_state=0).fit(SELarousal_train, y_train).predict(SELarousal_test)
    pred[:,1] = predictions
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = RandomForestClassifier(random_state=0).fit(SELvalence_train, y_train).predict(SELvalence_test)
    pred[:,2] = predictions
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = RandomForestClassifier(random_state=0).fit(SELall_train, y_train).predict(SELall_test)
    pred[:,3] = predictions
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred


def MLP(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall):
    pred = np.zeros((len(y_test),4))
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=8, hidden_layer_sizes=(25,25)).fit(x_train, y_train).predict(x_test)
    pred[:,0] = predictions
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=8, hidden_layer_sizes=(25,25)).fit(SELarousal_train, y_train).predict(SELarousal_test)
    pred[:,1] = predictions
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=8, hidden_layer_sizes=(25,25)).fit(SELvalence_train, y_train).predict(SELvalence_test)
    pred[:,2] = predictions
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=8, hidden_layer_sizes=(25,25)).fit(SELall_train, y_train).predict(SELall_test)
    pred[:,3] = predictions
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    print(SELall_test.shape[1], ' features in A+V set')
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred


def evaluation(predictions, y_test, mean_UAR, mean_WAR, mean_cm):
    UAR = recall_score(y_test, predictions, average='macro')
    mean_UAR = mean_UAR + UAR
    WAR = recall_score(y_test, predictions, average='weighted')
    mean_WAR = mean_WAR + WAR
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4])
    mean_cm = np.add(mean_cm, cm)
    return mean_UAR, mean_WAR, mean_cm


def add_average_results(average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall, mean_UAR_all, mean_cm_all, mean_UAR_A, mean_cm_A, mean_UAR_V, mean_cm_V, mean_UAR_SELall, mean_cm_SELall):
    average_UAR_all = average_UAR_all + mean_UAR_all
    average_UAR_A = average_UAR_A + mean_UAR_A
    average_UAR_V = average_UAR_V + mean_UAR_V
    average_UAR_SELall = average_UAR_SELall + mean_UAR_SELall
    average_mean_cm_all = np.add(average_mean_cm_all, mean_cm_all)
    average_mean_cm_A = np.add(average_mean_cm_A, mean_cm_A)
    average_mean_cm_V = np.add(average_mean_cm_V, mean_cm_V)
    average_mean_cm_SELall = np.add(average_mean_cm_SELall, mean_cm_SELall)
    return average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall


def print_function(classifier, mean_UAR_all, mean_cm_all, mean_UAR_A, mean_cm_A, mean_UAR_V, mean_cm_V, mean_UAR_SELall, mean_cm_SELall, N):
    print("---------------------------")
    print(classifier)
    print("(all features)")
    print("UAR", mean_UAR_all/N)
    CM = mean_cm_all/N
    new_CM = np.true_divide(CM, CM.sum(axis=1, keepdims=True))
    new_CM = new_CM*100
    print(np.around(new_CM, decimals=1, out=None))
    print("(selected for Arousal modelling)")
    print("UAR", mean_UAR_A/N)
    CM = mean_cm_A/N
    new_CM = np.true_divide(CM, CM.sum(axis=1, keepdims=True))
    new_CM = new_CM*100
    print(np.around(new_CM, decimals=1, out=None))
    print("(selected for Valence modelling)")
    print("UAR", mean_UAR_V/N)
    CM = mean_cm_V/N
    new_CM = np.true_divide(CM, CM.sum(axis=1, keepdims=True))
    new_CM = new_CM*100
    print(np.around(new_CM, decimals=1, out=None))
    print("(selected for Valence & Arousal togethere)")
    print("UAR", mean_UAR_SELall/N)
    CM = mean_cm_SELall/N
    new_CM = np.true_divide(CM, CM.sum(axis=1, keepdims=True))
    new_CM = new_CM*100
    print(np.around(new_CM, decimals=1, out=None))


if __name__ == "__main__":
    my_dir = os.getcwd()
    df, A_mean, V_mean, A_EWE, V_EWE, y, x, A_selection_index, V_selection_index, all_selection_index = get_data(my_dir)
    average_UAR_all = 0
    average_mean_cm_all = np.zeros((5,5), dtype=np.int)
    average_UAR_V = 0
    average_mean_cm_V = np.zeros((5,5), dtype=np.int)
    average_UAR_A = 0
    average_mean_cm_A = np.zeros((5,5), dtype=np.int)
    average_UAR_SELall = 0
    average_mean_cm_SELall = np.zeros((5,5), dtype=np.int)
    for rand in list(range(5)):
        x_train, x_test, y_train, y_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, SELall_test, SELall_train = partitioning(x, y, A_selection_index, V_selection_index, all_selection_index, rand)
        majority_vote_results = np.zeros((y_test.shape[0],4,4), dtype=np.int)
        for classifier in ["SVM", "knn", "RF", "MLP"]:
            some_descriptive(x, y)
            mean_UAR_all = 0
            mean_WAR_all = 0
            mean_cm_all = np.zeros((5,5), dtype=np.int)
            mean_UAR_A = 0
            mean_WAR_A = 0
            mean_cm_A = np.zeros((5,5), dtype=np.int)
            mean_UAR_V = 0
            mean_WAR_V = 0
            mean_cm_V = np.zeros((5,5), dtype=np.int)
            mean_UAR_SELall = 0
            mean_WAR_SELall = 0
            mean_cm_SELall = np.zeros((5,5), dtype=np.int)
            if classifier == "SVM":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = SVM(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
                majority_vote_results[:,:,0] = pred
            elif classifier == "knn":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = knn(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
                majority_vote_results[:,:,1] = pred
            elif classifier == "RF":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = RF(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
                majority_vote_results[:,:,2] = pred
            elif classifier == "MLP":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = MLP(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
                majority_vote_results[:,:,3] = pred

        majority_vote_results = scipy.stats.mode(majority_vote_results, axis=2)[0].reshape(-1,4)

        mean_UAR_all = 0
        mean_WAR_all = 0
        mean_cm_all = np.zeros((5,5), dtype=np.int)
        mean_UAR_A = 0
        mean_WAR_A = 0
        mean_cm_A = np.zeros((5,5), dtype=np.int)
        mean_UAR_V = 0
        mean_WAR_V = 0
        mean_cm_V = np.zeros((5,5), dtype=np.int)
        mean_UAR_SELall = 0
        mean_WAR_SELall = 0
        mean_cm_SELall = np.zeros((5,5), dtype=np.int)

        mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(majority_vote_results[:,0], y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
        mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(majority_vote_results[:,1], y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
        mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(majority_vote_results[:,2], y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
        mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(majority_vote_results[:,3], y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)

        average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall = add_average_results(average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall, mean_UAR_all, mean_cm_all, mean_UAR_A, mean_cm_A, mean_UAR_V, mean_cm_V, mean_UAR_SELall, mean_cm_SELall)
    classifier = "MAJORITY"
    print_function(classifier, average_UAR_all, average_mean_cm_all, average_UAR_A, average_mean_cm_A, average_UAR_V, average_mean_cm_V, average_UAR_SELall, average_mean_cm_SELall, 5)

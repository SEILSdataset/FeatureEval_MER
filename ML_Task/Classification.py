import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network
from sklearn import tree
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
    y = df.iloc[:-1, 7:8].values
    x = df.iloc[:-1, 8:]
    A_selection = ["Most_Common_Rhythmic_Value", "Number_of_Strong_Rhythmic_Pulses_._Tempo_Standardized", "Median_Rhythmic_Value_Offset", "pcm_zcr_sma_skewness", "pcm_zcr_sma_stddev", "F0env_sma_linregerrQ", "Non.Standard_Chords"]
    A_selection_index = [x.columns.get_loc(c) for c in A_selection if c in x]  # get indexes of selected features
    V_selection = ["Minor_Major_Triad_Ratio", "pcm_zcr_sma_skewness", "F0env_sma_skewness", "F0env_sma_linregerrQ", "Amount_of_Staccato", "Strength_Ratio_of_Two_Strongest_Rhythmic_Pulses_._Tempo_Standardized", "BPM"]
    V_selection_index = [x.columns.get_loc(c) for c in V_selection if c in x]
    all_selection_index = A_selection_index + V_selection_index
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
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(x_train, y_train).predict(x_test)
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(SELarousal_train, y_train).predict(SELarousal_test)
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(SELvalence_train, y_train).predict(SELvalence_test)
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = OneVsRestClassifier(svm.LinearSVC(random_state=0)).fit(SELall_train, y_train).predict(SELall_test)
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall


def knn(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall):
    predictions = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train).predict(x_test)
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = KNeighborsClassifier(n_neighbors=5).fit(SELarousal_train, y_train).predict(SELarousal_test)
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = KNeighborsClassifier(n_neighbors=5).fit(SELvalence_train, y_train).predict(SELvalence_test)
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = KNeighborsClassifier(n_neighbors=5).fit(SELall_train, y_train).predict(SELall_test)
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall


def RF(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall):
    predictions = tree.DecisionTreeClassifier(random_state=0, criterion='gini').fit(x_train, y_train).predict(x_test)
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = tree.DecisionTreeClassifier(random_state=0, criterion='gini').fit(SELarousal_train, y_train).predict(SELarousal_test)
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = tree.DecisionTreeClassifier(random_state=0, criterion='gini').fit(SELvalence_train, y_train).predict(SELvalence_test)
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = tree.DecisionTreeClassifier(random_state=0, criterion='gini').fit(SELall_train, y_train).predict(SELall_test)
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall


def MLP(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall):
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=1, hidden_layer_sizes=(15,)).fit(x_train, y_train).predict(x_test)
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=1, hidden_layer_sizes=(15,)).fit(SELarousal_train, y_train).predict(SELarousal_test)
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=1, hidden_layer_sizes=(15,)).fit(SELvalence_train, y_train).predict(SELvalence_test)
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = neural_network.MLPClassifier(random_state=0, batch_size=1, hidden_layer_sizes=(15,)).fit(SELall_train, y_train).predict(SELall_test)
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall


def evaluation(predictions, y_test, mean_UAR, mean_WAR, mean_cm):
    UAR = recall_score(y_test, predictions, average='macro')
    mean_UAR = mean_UAR + UAR
    WAR = recall_score(y_test, predictions, average='weighted')
    mean_WAR = mean_WAR + WAR
    cm = confusion_matrix(y_test, predictions, labels=['Joy', 'Dysphoria', 'Sadness', 'Tranquility', 'Activation'])
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
    print("N. of samples: 2 Activation, 1 Dysphoria, 2 Joy, 1 Sadness, 1 Tranquility")
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
    average_mean_cm_all = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    average_UAR_V = 0
    average_mean_cm_V = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    average_UAR_A = 0
    average_mean_cm_A = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    average_UAR_SELall = 0
    average_mean_cm_SELall = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    for classifier in ["SVM", "knn", "RF", "MLP"]:
        some_descriptive(x, y)
        mean_UAR_all = 0
        mean_WAR_all = 0
        mean_cm_all = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        mean_UAR_A = 0
        mean_WAR_A = 0
        mean_cm_A = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        mean_UAR_V = 0
        mean_WAR_V = 0
        mean_cm_V = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        mean_UAR_SELall = 0
        mean_WAR_SELall = 0
        mean_cm_SELall = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        for rand in list(range(5)):
            x_train, x_test, y_train, y_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, SELall_test, SELall_train = partitioning(x, y, A_selection_index, V_selection_index, all_selection_index, rand)
            if classifier == "SVM":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = SVM(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
            elif classifier == "knn":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = knn(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
            elif classifier == "RF":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = RF(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
            elif classifier == "MLP":
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = MLP(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
        average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall = add_average_results(average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall, mean_UAR_all, mean_cm_all, mean_UAR_A, mean_cm_A, mean_UAR_V, mean_cm_V, mean_UAR_SELall, mean_cm_SELall)
        print_function(classifier, mean_UAR_all, mean_cm_all, mean_UAR_A, mean_cm_A, mean_UAR_V, mean_cm_V, mean_UAR_SELall, mean_cm_SELall, 5)
    classifier = "ALL MODELS"
    print_function(classifier, average_UAR_all, average_mean_cm_all, average_UAR_A, average_mean_cm_A, average_UAR_V, average_mean_cm_V, average_UAR_SELall, average_mean_cm_SELall, 20)

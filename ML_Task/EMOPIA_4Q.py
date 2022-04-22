import numpy as np
import pandas as pd
import os
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn.model_selection import GroupKFold
from operator import itemgetter
import warnings
warnings.filterwarnings('ignore')


def get_data(my_dir):
    df = pd.read_csv(my_dir + '/EMOPIA_features.csv')
    labels = {"target": [], "song": [], "clip": []}
    for index, row in df.iterrows():
        file_name = row['song']
        target, b = file_name.split('_', 1)
        labels["target"].append(target)
        name, clip = b.rsplit('_', 1)
        labels["song"].append(name)
        labels["clip"].append(clip)
    labels_df = pd.DataFrame.from_dict(labels)
    y_col = labels_df.iloc[:, :1]
    y_col = y_col.replace('Q1', 0)
    y_col = y_col.replace('Q2', 1)
    y_col = y_col.replace('Q3', 2)
    y_col = y_col.replace('Q4', 3)
    y = y_col.values
    x = df.iloc[:, 1:]
    A_selection_Cor = ["Most_Common_Rhythmic_Value", "Number_of_Strong_Rhythmic_Pulses_._Tempo_Standardized", "BPM", "pcm_zcr_sma_skewness", "Note_Density", "Standard_Triads", "Melodic_Large_Intervals", "Most_Common_Vertical_Interval", "Strength_Ratio_of_Two_Strongest_Rhythmic_Pulses_._Tempo_Standardized", "Prevalence_of_Dotted_Notes"]
    V_selection_Cor = ["F0env_sma_quartile3", "Minor_Major_Triad_Ratio", "Minor_Major_Melodic_Third_Ratio", "F0env_sma_skewness", "pcm_intensity_sma_minPos", "Similar_Motion", "Strength_Ratio_of_Two_Strongest_Rhythmic_Pulses_._Tempo_Standardized", "Diminished_and_Augmented_Triads", "Dynamic_Range", "Vertical_Perfect_Fifths"]
    A_selection_LR = ["Most_Common_Rhythmic_Value", "Number_of_Strong_Rhythmic_Pulses_._Tempo_Standardized", "Median_Rhythmic_Value_Offset", "pcm_zcr_sma_skewness", "pcm_zcr_sma_stddev", "F0env_sma_linregerrQ"]
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
    return df, labels_df, y, x, A_selection_index, V_selection_index, all_selection_index


def some_descriptive(x, y, split):
    print("---------------------------")
    print(split)
    print(x.shape[0], 'instances from', len(np.unique(y)), 'classes', np.unique(y))
    for q in [0, 1, 2, 3]:
        count = np.count_nonzero(y == q)
        print(str(count), 'of:', q)
    print(x.shape[1], 'features')


def partitioning(x, y, labels_df, A_selection_index, V_selection_index, all_selection_index, rand):
    group_kfold = GroupKFold(n_splits=5)
    group_kfold2 = GroupKFold(n_splits=2)
    x = x.values
    group = labels_df["song"].values
    iter = 0

    for train_index, test_index in group_kfold.split(x, y, group):
        if iter <= rand:
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            group_train, group_test = group[train_index], group[test_index]

            train_index, dev_index = next(group_kfold2.split(x_train, y_train, group_train))
            x_train, x_dev = x_train[train_index], x_train[dev_index]
            y_train, y_dev = y_train[train_index], y_train[dev_index]
            iter += 1

    scaler = StandardScaler()  # normalising features in the training set
    x_train = scaler.fit_transform(x_train)
    x_dev = scaler.transform(x_dev)  # Applies the same scaling and shifting operations performed on the train data
    x_test = scaler.transform(x_test)  # Applies the same scaling and shifting operations performed on the train data
    some_descriptive(x_train, y_train, 'TRAINING SET')
    some_descriptive(x_dev, y_dev, 'DEV SET')
    some_descriptive(x_test, y_test, 'TEST SET')
    SELarousal_test = np.take(x_test, A_selection_index, 1)  # filter out array according to the indexes of selected features
    SELarousal_train = np.take(x_train, A_selection_index, 1)
    SELarousal_dev = np.take(x_dev, A_selection_index, 1)
    SELvalence_test = np.take(x_test, V_selection_index, 1)
    SELvalence_train = np.take(x_train, V_selection_index, 1)
    SELvalence_dev = np.take(x_dev, V_selection_index, 1)
    SELall_test = np.take(x_test, all_selection_index, 1)
    SELall_train = np.take(x_train, all_selection_index, 1)
    SELall_dev = np.take(x_dev, all_selection_index, 1)
    return x_train, x_test, y_train, y_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, SELall_test, SELall_train, SELarousal_dev, SELvalence_dev, SELall_dev, x_dev, y_dev


def get_clf(classifier, elem):
    if classifier == 'SVM':
        clf = svm.SVC(C=elem, kernel='linear')
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=elem)
    elif classifier == 'RF':
        clf = RandomForestClassifier(random_state=0, n_estimators=elem)
    elif classifier == 'MLP':
        clf = neural_network.MLPClassifier(random_state=0, batch_size=8, hidden_layer_sizes=(elem,elem))
    return clf


def run_experiments(X_train, Y_train, X_dev, Y_dev, X_test, opt, classifier):
    #  OPTIMISATION  #
    results = []
    for elem in opt:
        clf = get_clf(classifier, elem)
        # Run model for each optimisation configuration
        clf.fit(X_train, Y_train.ravel())
        predictions = clf.predict(X_dev)
        UAR = recall_score(Y_dev, predictions, average='macro')
        results.append((UAR, elem))

    #  TEST  #
    best = max(results, key=itemgetter(0))[1]
    # Merge train and dev to perform the final training with more data
    X_train = np.concatenate((X_train, X_dev), axis=0)
    Y_train = np.concatenate((Y_train, Y_dev), axis=0)
    # Make training again with the optimal hyper-parameters
    clf = get_clf(classifier, best)
    clf.fit(X_train, Y_train)
    # Make final test
    predictions = clf.predict(X_test)
    return predictions


def ML(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, SELarousal_dev, SELvalence_dev, SELall_dev, x_dev, y_dev, opt, classifier):
    pred = np.zeros((len(y_test),4))
    predictions = run_experiments(x_train, y_train, x_dev, y_dev, x_test, opt, classifier)
    pred[:,0] = predictions
    mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(predictions, y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
    predictions = run_experiments(SELarousal_train, y_train, SELarousal_dev, y_dev, SELarousal_test, opt, classifier)
    pred[:,1] = predictions
    mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(predictions, y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
    predictions = run_experiments(SELvalence_train, y_train, SELvalence_dev, y_dev, SELvalence_test, opt, classifier)
    pred[:,2] = predictions
    mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(predictions, y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
    predictions = run_experiments(SELall_train, y_train, SELall_dev, y_dev, SELall_test, opt, classifier)
    pred[:,3] = predictions
    mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(predictions, y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)
    return mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred


def evaluation(predictions, y_test, mean_UAR, mean_WAR, mean_cm):
    UAR = recall_score(y_test, predictions, average='macro')
    mean_UAR = mean_UAR + UAR
    WAR = recall_score(y_test, predictions, average='weighted')
    mean_WAR = mean_WAR + WAR
    cm = confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3])
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
    df, labels_df, y, x, A_selection_index, V_selection_index, all_selection_index = get_data(my_dir)
    average_UAR_all = 0
    average_mean_cm_all = np.zeros((4,4), dtype=np.int)
    average_UAR_V = 0
    average_mean_cm_V = np.zeros((4,4), dtype=np.int)
    average_UAR_A = 0
    average_mean_cm_A = np.zeros((4,4), dtype=np.int)
    average_UAR_SELall = 0
    average_mean_cm_SELall = np.zeros((4,4), dtype=np.int)
    for rand in list(range(5)):
        x_train, x_test, y_train, y_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, SELall_test, SELall_train, SELarousal_dev, SELvalence_dev, SELall_dev, x_dev, y_dev = partitioning(x, y, labels_df, A_selection_index, V_selection_index, all_selection_index, rand)
        majority_vote_results = np.zeros((y_test.shape[0],4,4), dtype=np.int)
        for classifier in ['SVM', 'knn', 'RF', 'MLP']:
            some_descriptive(x, y, 'EMOPIA DATA')
            mean_UAR_all = 0
            mean_WAR_all = 0
            mean_cm_all = np.zeros((4,4), dtype=np.int)
            mean_UAR_A = 0
            mean_WAR_A = 0
            mean_cm_A = np.zeros((4,4), dtype=np.int)
            mean_UAR_V = 0
            mean_WAR_V = 0
            mean_cm_V = np.zeros((4,4), dtype=np.int)
            mean_UAR_SELall = 0
            mean_WAR_SELall = 0
            mean_cm_SELall = np.zeros((4,4), dtype=np.int)
            if classifier == "SVM":
                opt = [0.0001, 0.001, 0.01, 0.1, 1.0]
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = ML(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, SELarousal_dev, SELvalence_dev, SELall_dev, x_dev, y_dev, opt, classifier)
                majority_vote_results[:,:,0] = pred
            elif classifier == "knn":
                opt = [3, 5, 7, 9, 11]
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = ML(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, SELarousal_dev, SELvalence_dev, SELall_dev, x_dev, y_dev, opt, classifier)
                majority_vote_results[:,:,1] = pred
            elif classifier == "RF":
                opt = [10, 50, 100, 150, 200]
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = ML(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, SELarousal_dev, SELvalence_dev, SELall_dev, x_dev, y_dev, opt, classifier)
                majority_vote_results[:,:,2] = pred
            elif classifier == "MLP":
                opt = [25, 50, 100, 175, 300]
                mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, pred = ML(x_train, y_train, x_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train, mean_UAR_all, mean_WAR_all, mean_cm_all, mean_UAR_A, mean_WAR_A, mean_cm_A, mean_UAR_V, mean_WAR_V, mean_cm_V, SELall_test, SELall_train, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall, SELarousal_dev, SELvalence_dev, SELall_dev, x_dev, y_dev, opt, classifier)
                majority_vote_results[:,:,3] = pred

        majority_vote_results = scipy.stats.mode(majority_vote_results, axis=2)[0].reshape(-1,4)

        mean_UAR_all = 0
        mean_WAR_all = 0
        mean_cm_all = np.zeros((4,4), dtype=np.int)
        mean_UAR_A = 0
        mean_WAR_A = 0
        mean_cm_A = np.zeros((4,4), dtype=np.int)
        mean_UAR_V = 0
        mean_WAR_V = 0
        mean_cm_V = np.zeros((4,4), dtype=np.int)
        mean_UAR_SELall = 0
        mean_WAR_SELall = 0
        mean_cm_SELall = np.zeros((4,4), dtype=np.int)

        mean_UAR_all, mean_WAR_all, mean_cm_all = evaluation(majority_vote_results[:,0], y_test, mean_UAR_all, mean_WAR_all, mean_cm_all)
        mean_UAR_A, mean_WAR_A, mean_cm_A = evaluation(majority_vote_results[:,1], y_test, mean_UAR_A, mean_WAR_A, mean_cm_A)
        mean_UAR_V, mean_WAR_V, mean_cm_V = evaluation(majority_vote_results[:,2], y_test, mean_UAR_V, mean_WAR_V, mean_cm_V)
        mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall = evaluation(majority_vote_results[:,3], y_test, mean_UAR_SELall, mean_WAR_SELall, mean_cm_SELall)

        average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall = add_average_results(average_UAR_all, average_mean_cm_all, average_UAR_V, average_mean_cm_V, average_UAR_A, average_mean_cm_A, average_UAR_SELall, average_mean_cm_SELall, mean_UAR_all, mean_cm_all, mean_UAR_A, mean_cm_A, mean_UAR_V, mean_cm_V, mean_UAR_SELall, mean_cm_SELall)
    classifier = "MAJORITY"
    print_function(classifier, average_UAR_all, average_mean_cm_all, average_UAR_A, average_mean_cm_A, average_UAR_V, average_mean_cm_V, average_UAR_SELall, average_mean_cm_SELall, 5)

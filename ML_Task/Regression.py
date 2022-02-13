import numpy as np
import pandas as pd
import os

import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
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
    return df, A_mean, V_mean, A_EWE, V_EWE, y, x, A_selection_index, V_selection_index


def partitioning(x, y, A_selection_index, V_selection_index, rand):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rand, stratify=y)
    scaler = StandardScaler()  # normalising features in the training set
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  # Applies the same scaling and shifting operations performed on the train data
    y_train = y_train.flatten()  # get correct dimension (1D array)
    y_test = y_test.flatten()
    SELarousal_test = np.take(x_test, A_selection_index, 1)  # filter out array according to the indexes of selected features
    SELarousal_train = np.take(x_train, A_selection_index, 1)
    SELvalence_test = np.take(x_test, V_selection_index, 1)
    SELvalence_train = np.take(x_train, V_selection_index, 1)
    return x_train, x_test, y_train, y_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train


def partitioning_reg(y_dim, y, rand):
    # split the gold standard according to y (for comparability in the partitioning wrt the classification task)
    y_train, y_test, n1, n2 = train_test_split(y_dim, y, test_size=0.3, random_state=rand, stratify=y)
    y_train = y_train.flatten()  # get correct dimension (1D array)
    y_test = y_test.flatten()
    return y_train, y_test


def run_experiments(x, y, A_selection_index, V_selection_index, all_scores, regr):
    for rand in list(range(5)):
        x_train, x_test, y_train, y_test, SELarousal_test, SELarousal_train, SELvalence_test, SELvalence_train = partitioning(x, y, A_selection_index, V_selection_index, rand)
        all_scores = run_regression(V_EWE, y, rand, x_train, x_test, 'Rsqr_EWE_V', 'mse_EWE_V', all_scores, regr)
        all_scores = run_regression(V_mean, y, rand, x_train, x_test, 'Rsqr_mean_V', 'mse_mean_V', all_scores, regr)
        all_scores = run_regression(V_EWE, y, rand, SELvalence_train, SELvalence_test, 'Rsqr_EWE_V_sel', 'mse_EWE_V_sel', all_scores, regr)
        all_scores = run_regression(V_mean, y, rand, SELvalence_train, SELvalence_test, 'Rsqr_mean_V_sel', 'mse_mean_V_sel', all_scores, regr)
        all_scores = run_regression(A_EWE, y, rand, x_train, x_test, 'Rsqr_EWE_A', 'mse_EWE_A', all_scores, regr)
        all_scores = run_regression(A_mean, y, rand, x_train, x_test, 'Rsqr_mean_A', 'mse_mean_A', all_scores, regr)
        all_scores = run_regression(A_EWE, y, rand, SELarousal_train, SELarousal_test, 'Rsqr_EWE_A_sel', 'mse_EWE_A_sel', all_scores, regr)
        all_scores = run_regression(A_mean, y, rand, SELarousal_train, SELarousal_test, 'Rsqr_mean_A_sel', 'mse_mean_A_sel', all_scores, regr)
    print_results(all_scores, 5)


def run_regression(target, y, rand, x_train, x_test, Rsqr_var, mse_var, all_scores, regr):
    y_train, y_test = partitioning_reg(target, y, rand)
    regr.fit(x_train, y_train)
    prediction = regr.score(x_test, y_test)
    prediction2 = regr.predict(x_test)
    mse = sklearn.metrics.mean_squared_error(y_test, prediction2)
    all_scores[Rsqr_var] = all_scores[Rsqr_var] + prediction
    all_scores[mse_var] = all_scores[mse_var] + mse
    return all_scores


def print_results(all_scores, N):
    print("Coefficient of Determination R(sqr)")
    print("EWE V = " + str(all_scores['Rsqr_EWE_V']/N))
    print("mean V = " + str(all_scores['Rsqr_mean_V']/N))
    print("EWE V (selected) = " + str(all_scores['Rsqr_EWE_V_sel']/N))
    print("mean V (selected) = " + str(all_scores['Rsqr_mean_V_sel']/N))
    print("EWE A = " + str(all_scores['Rsqr_EWE_A']/N))
    print("mean A = " + str(all_scores['Rsqr_mean_A']/N))
    print("EWE A (selected) = " + str(all_scores['Rsqr_EWE_A_sel']/N))
    print("mean A (selected) = " + str(all_scores['Rsqr_mean_A_sel']/N))
    print("Mean Square Error")
    print("EWE V = " + str(all_scores['mse_EWE_V']/N))
    print("mean V = " + str(all_scores['mse_mean_V']/N))
    print("EWE V (selected) = " + str(all_scores['mse_EWE_V_sel']/N))
    print("mean V (selected) = " + str(all_scores['mse_mean_V_sel']/N))
    print("EWE A = " + str(all_scores['mse_EWE_A']/N))
    print("mean A = " + str(all_scores['mse_mean_A']/N))
    print("EWE A (selected) = " + str(all_scores['mse_EWE_A_sel']/N))
    print("mean A (selected) = " + str(all_scores['mse_mean_A_sel']/N))
    print("--------------------------------------------------")


def add_results(Final_results, all_scores):
    for key in all_scores:
        Final_results[key] = Final_results[key] + all_scores[key]
    return Final_results


if __name__ == "__main__":
    my_dir = os.getcwd()
    df, A_mean, V_mean, A_EWE, V_EWE, y, x, A_selection_index, V_selection_index = get_data(my_dir)
    Final_results = {'Rsqr_EWE_V': 0, 'Rsqr_mean_V': 0, 'mse_EWE_V': 0, 'mse_mean_V': 0,
                  'Rsqr_EWE_V_sel': 0, 'Rsqr_mean_V_sel': 0, 'mse_EWE_V_sel': 0, 'mse_mean_V_sel': 0,
                  'Rsqr_EWE_A': 0, 'Rsqr_mean_A': 0, 'mse_EWE_A': 0, 'mse_mean_A': 0,
                  'Rsqr_EWE_A_sel': 0, 'Rsqr_mean_A_sel': 0, 'mse_EWE_A_sel': 0, 'mse_mean_A_sel': 0}
    for regressor in ["SVM", "knn", "RF", "MLP"]:
        all_scores = {'Rsqr_EWE_V': 0, 'Rsqr_mean_V': 0, 'mse_EWE_V': 0, 'mse_mean_V': 0,
                      'Rsqr_EWE_V_sel': 0, 'Rsqr_mean_V_sel': 0, 'mse_EWE_V_sel': 0, 'mse_mean_V_sel': 0,
                      'Rsqr_EWE_A': 0, 'Rsqr_mean_A': 0, 'mse_EWE_A': 0, 'mse_mean_A': 0,
                      'Rsqr_EWE_A_sel': 0, 'Rsqr_mean_A_sel': 0, 'mse_EWE_A_sel': 0, 'mse_mean_A_sel': 0}
        if regressor == "SVM":
            print("** RESULTS FOR SVM **")
            regr = svm.SVR(kernel='linear')
            run_experiments(x, y, A_selection_index, V_selection_index, all_scores, regr)
            Final_results = add_results(Final_results, all_scores)
        elif regressor == "knn":
            print("** RESULTS FOR knn **")
            regr = KNeighborsRegressor(n_neighbors=5)
            run_experiments(x, y, A_selection_index, V_selection_index, all_scores, regr)
            Final_results = add_results(Final_results, all_scores)
        elif regressor == "MLP":
            print("** RESULTS FOR MLP **")
            regr = MLPRegressor(random_state=0, batch_size=1, hidden_layer_sizes=(15,))
            run_experiments(x, y, A_selection_index, V_selection_index, all_scores, regr)
            Final_results = add_results(Final_results, all_scores)
        else:
            print("** RESULTS FOR RF **")
            regr = RandomForestRegressor(max_depth=2, random_state=0)
            run_experiments(x, y, A_selection_index, V_selection_index, all_scores, regr)
            Final_results = add_results(Final_results, all_scores)
    print("** AVERAGE RESULTS ACROSS MODELS **")
    print_results(Final_results, 20)



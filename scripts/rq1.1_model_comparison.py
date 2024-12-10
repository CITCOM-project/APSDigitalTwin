from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.model import Model
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view 
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import multiprocessing as mp
from scipy import stats

from gplearn.genetic import SymbolicRegressor
from sklearn.neural_network import MLPRegressor

def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def process(data_trace):
    
    training_data = TrainingData(os.path.join("./data", data_trace))
    np_bg_training = np.array(training_data.bg_data_frame)
    np_cob_training = np.array(training_data.cob_data_frame)
    np_iob_training = np.array(training_data.iob_data_frame)
    
    bg_pairs = sliding_window_view(np_bg_training, 2)
    cob_pairs = sliding_window_view(np_cob_training, 2)
    iob_pairs = sliding_window_view(np_iob_training, 2)
    
    train_x = []
    train_y = []
    train_y_g = []
    train_y_s = []
    train_y_i = []
    
    for i in range(len(bg_pairs)):
        train_x.append([bg_pairs[i][0], cob_pairs[i][0], iob_pairs[i][0]])
        train_y.append([bg_pairs[i][1], cob_pairs[i][1], iob_pairs[i][1]])
        train_y_g.append(bg_pairs[i][1])
        train_y_s.append(cob_pairs[i][1])
        train_y_i.append(iob_pairs[i][1])
    
    sr_g = SymbolicRegressor()
    sr_s = SymbolicRegressor()
    sr_i = SymbolicRegressor()
    sr_g.fit(train_x, train_y_g)
    sr_s.fit(train_x, train_y_s)
    sr_i.fit(train_x, train_y_i)
    
    history = []
    history.append({'step': 0, s_label: training_data.find_initial_values()[0], g_label: training_data.find_initial_values()[3], i_label: training_data.find_initial_values()[4]})
    for i in range(1, (training_data.timesteps - 1) * 5 + 1):
        
        features = [((history[i - 1][g_label]), (history[i - 1][s_label]), (history[i - 1][i_label]))]
        new_g = min(sr_g.predict(features)[0], np.finfo(np.float16).max)
        new_s = min(sr_s.predict(features)[0], np.finfo(np.float16).max)
        new_i = min(sr_i.predict(features)[0], np.finfo(np.float16).max)
        
        for intervention in training_data.interventions:
            if i == intervention[0]:
                if intervention[1] == s_label:
                    new_s += intervention[2]
                elif intervention[1] == i_label:
                    new_i += intervention[2]
        
        history.append({'step': i, s_label: new_s, g_label: new_g, i_label: new_i})
    
    np_bg_model = np.array(pd.DataFrame(history)[g_label])
    training_error_sr = math.sqrt((1 / len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_training)))
    
    nn = MLPRegressor(max_iter=1000)
    nn.fit(train_x, train_y)
    
    history = []
    history.append({'step': 0, s_label: training_data.find_initial_values()[0], g_label: training_data.find_initial_values()[3], i_label: training_data.find_initial_values()[4]})
    for i in range(1, (training_data.timesteps - 1) * 5 + 1):
        
        features = [((history[i - 1][g_label]), (history[i - 1][s_label]), (history[i - 1][i_label]))]
        res = nn.predict(features)[0]
        new_g = res[0]
        new_s = res[1]
        new_i = res[2]
        
        for intervention in training_data.interventions:
            if i == intervention[0]:
                if intervention[1] == s_label:
                    new_s += intervention[2]
                elif intervention[1] == i_label:
                    new_i += intervention[2]
        
        history.append({'step': i, s_label: new_s, g_label: new_g, i_label: new_i})
    
    np_bg_model = np.array(pd.DataFrame(history)[g_label])
    training_error_nn = math.sqrt((1 / len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_training)))
    
    ga = GlucoseInsulinGeneticAlgorithm()
    constants, _f = ga.run(training_data)
    training_model = Model(training_data.find_initial_values(), constants)

    for intervention in training_data.interventions:
        training_model.add_intervention(intervention[0], intervention[1], intervention[2])

    for i in range(1, (training_data.timesteps - 1) * 5 + 1):
        training_model.update(i)

    np_bg_model = np.array(pd.DataFrame(training_model.history)[g_label])
    training_error_model = math.sqrt((1 / len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_training)))
    
    print(training_error_sr)
    print(training_error_nn)
    print(training_error_model)
    
    return training_error_model, training_error_sr, training_error_nn
    
if __name__ == "__main__":
    figure_output = "<path>"
    
    all_traces = os.listdir("./data")

    all_model_fitnesses = []
    all_sr_fitnesses = []
    all_nn_fitnesses = []
    
    num = 1
    with mp.Pool(processes=num) as pool:
        pool_vals = []
        for data_trace in all_traces:
            if data_trace.endswith(".csv"):
                pool_vals.append(data_trace)
        
        res_list = pool.map(process, pool_vals)
        
        for training_error_model, training_error_sr, training_error_nn in res_list:
            all_model_fitnesses.append(training_error_model)
            all_sr_fitnesses.append(training_error_sr)
            all_nn_fitnesses.append(training_error_nn)
                
    data = pd.DataFrame([all_model_fitnesses, all_sr_fitnesses, all_nn_fitnesses])
    data.to_csv("comparison_data.csv")
    
    data = pd.read_csv("comparison_data.csv").transpose()
    data.columns = ["Adapted Model", "Symbolic Regressor", "Neural Network"]
    plt.boxplot(data, showfliers=False, labels=["Adapted Model", "Symbolic Regressor", "Neural Network"])
    plt.ylabel("RMSE", fontsize=12)
    plt.title("Model Comparison")
    plt.gca().tick_params(axis='x', labelsize=11)
    plt.gca().tick_params(axis='y', labelsize=11)
    plt.savefig(os.path.join(figure_output, "RQ1.1.png"))
    plt.show()

    data_am = data["Adapted Model"]
    data_sr = data["Symbolic Regressor"]
    data_nn = data["Neural Network"]

    data_am_r = [x for x in data["Adapted Model"] if x < 200]
    data_sr_r = [x for x in data["Symbolic Regressor"] if x < 200]
    data_nn_r = [x for x in data["Neural Network"] if x < 200]

    print(np.mean(data_am))
    print(np.mean(data_sr))
    print(np.mean(data_nn))
    print()
    print(stats.shapiro(data_am).pvalue)
    print(stats.shapiro(data_sr).pvalue)
    print(stats.shapiro(data_nn).pvalue)
    print()
    print(stats.mannwhitneyu(data_am, data_sr).pvalue)
    print(stats.mannwhitneyu(data_am, data_nn).pvalue)
    print()
    print(cohen_d(data_am, data_sr))
    print(cohen_d(data_am, data_nn))
    print()
    print(np.mean(data_am_r))
    print(np.mean(data_sr_r))
    print(np.mean(data_nn_r))
    print()
    print(stats.shapiro(data_am_r).pvalue)
    print(stats.shapiro(data_sr_r).pvalue)
    print(stats.shapiro(data_nn_r).pvalue)
    print()
    print(stats.mannwhitneyu(data_am_r, data_sr_r).pvalue)
    print(stats.mannwhitneyu(data_am_r, data_nn_r).pvalue)
    print()
    print(cohen_d(data_am_r, data_sr_r))
    print(cohen_d(data_am_r, data_nn_r))
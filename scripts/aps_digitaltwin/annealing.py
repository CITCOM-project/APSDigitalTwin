import numpy as np
import pandas as pd
import warnings
import math
from scipy import optimize
from aps_digitaltwin.util import TrainingData, s_label, i_label, g_label
from aps_digitaltwin.model import Model

def fitness_function_stomach(solution, training_data, _x):
    training_data = training_data
    constants = [solution[0],0.1,0.1,0.1,0.1,0.1,0.1,5,0.1,0.1,0.1,0.1]
    error = 0

    model = Model(training_data.find_initial_values(), constants)

    for intervention in training_data.interventions:
        model.add_intervention(intervention[0], intervention[1], intervention[2])

    try:
        for i in range(1, (training_data.timesteps - 1) * 5 + 1):
            model.update(i)
    except:
        return 9999999999
    
    np_stomach_model = np.array(pd.DataFrame(model.history)[s_label])
    np_stomach_training = np.array(training_data.cob_data_frame)
    error += np.sum(np.square(np_stomach_model - np_stomach_training))

    return error

def fitness_function_insulin(solution, training_data, kjs):

    constants = [kjs,0.1,0.1,0.1,0.1,0.1,solution[0],5,0.1,0.1,0.1,0.1]
    error = 0

    model = Model(training_data.find_initial_values(), constants)

    for intervention in training_data.interventions:
        model.add_intervention(intervention[0], intervention[1], intervention[2])

    try:
        for i in range(1, (training_data.timesteps - 1) * 5 + 1):
            model.update(i)
    except:
        return 9999999999
    
    np_insulin_model = np.array(pd.DataFrame(model.history)[i_label])
    np_insulin_training = np.array(training_data.iob_data_frame)
    error += np.sum(np.square(np_insulin_model - np_insulin_training))

    return error

def fitness_function_glucose(solution, training_data, kjs, kxi):

    constants = [
        kjs,
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        kxi,
        int(solution[5]),
        solution[6],
        solution[7],
        solution[8],
        solution[9]
    ]
    error = 0
    
    model = Model(training_data.find_initial_values(), constants)

    for intervention in training_data.interventions:
        model.add_intervention(intervention[0], intervention[1], intervention[2])

    try:
        for i in range(1, (training_data.timesteps - 1) * 5 + 1):
            model.update(i)
    except Exception as e:
        return 9999999999

    np_bg_model = np.array(pd.DataFrame(model.history)[g_label])
    np_bg_training = np.array(training_data.bg_data_frame)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        spline_factor = 0.1
        error += math.sqrt((1/len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_training)))
        error += math.sqrt((1/len(np_bg_model)) * np.sum(np.square(np.diff(np_bg_model))) * spline_factor)

    if math.isinf(error) or math.isnan(error):
        return 9999999999

    return error
    
class GlucoseInsulinAnnealing:

    def run(self, training_data):

        res_1 = optimize.dual_annealing(fitness_function_stomach, bounds=[[0, 1]], args=(training_data,1), maxiter=100,
                                        no_local_search=True)
        
        res_2 = optimize.dual_annealing(fitness_function_insulin, bounds=[[0, 1]], args=(training_data, res_1.x[0]), maxiter=100,
                                        no_local_search=True)
        
        res_3 = optimize.dual_annealing(fitness_function_glucose, 
                                        bounds=[[0,1],[0,1],[0,1],[0,1],[0,1],[1,training_data.timesteps*5],[0,1],[0,1],[0,1],[0,10]], 
                                        args=(training_data, res_1.x[0], res_2.x[0]), maxiter=100, no_local_search=True)
        
        best_constants = [
            res_1.x[0],
            res_3.x[0],
            res_3.x[1],
            res_3.x[2],
            res_3.x[3],
            res_3.x[4],
            res_2.x[0],
            int(res_3.x[5]),
            res_3.x[6],
            res_3.x[7],
            res_3.x[8],
            res_3.x[9]
        ]

        return best_constants, 1/res_3.fun
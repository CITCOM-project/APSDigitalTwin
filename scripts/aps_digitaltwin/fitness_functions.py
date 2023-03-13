import numpy as np
import pandas as pd
import math
from aps_digitaltwin.model import Model
from aps_digitaltwin.util import s_label, i_label, g_label

def fitness_function_stomach(solutions, training_data):
    costs = []
    for solution in solutions:
        constants = [solution[0],0.1,0.1,0.1,0.1,0.1,0.1,5,0.1,0.1,0.1,0.1,0.1]

        model = Model(training_data.find_initial_values(), constants)

        for intervention in training_data.interventions:
            model.add_intervention(intervention[0], intervention[1], intervention[2])

        try:
            for i in range(1, (training_data.timesteps - 1) * 5 + 1):
                model.update(i)
        except:
            costs.append(99999999)
            continue
        
        np_stomach_model = np.array(pd.DataFrame(model.history)[s_label])
        np_stomach_training = np.array(training_data.cob_data_frame)
        error = np.sum(np.square(np_stomach_model - np_stomach_training))

        costs.append(error)
    
    return costs

def fitness_function_insulin(solutions, training_data, kjs):
    costs = []
    for solution in solutions:
        constants = [kjs,0.1,0.1,0.1,0.1,0.1,solution[0],5,0.1,0.1,0.1,0.1]

        model = Model(training_data.find_initial_values(), constants)

        for intervention in training_data.interventions:
            model.add_intervention(intervention[0], intervention[1], intervention[2])

        try:
            for i in range(1, (training_data.timesteps - 1) * 5 + 1):
                model.update(i)
        except:
            costs.append(99999999)
            continue
        
        np_insulin_model = np.array(pd.DataFrame(model.history)[i_label])
        np_insulin_training = np.array(training_data.iob_data_frame)

        spline_factor = 0.01
        error = np.sum(np.square(np_insulin_model - np_insulin_training))
        error += np.sum(np.square(np.diff(np_insulin_model))) * spline_factor

        costs.append(error)

    return costs

def fitness_function_glucose(solutions, training_data, kjs, kxi):
    costs = []
    for solution in solutions:
        constants = [
            kjs,
            solution[0],
            solution[1],
            solution[2],
            solution[3],
            solution[4],
            kxi,
            round(solution[5]),
            solution[6],
            solution[7],
            solution[8],
            solution[9]
        ]

        model = Model(training_data.find_initial_values(), constants)

        for intervention in training_data.interventions:
            model.add_intervention(intervention[0], intervention[1], intervention[2])

        try:
            for i in range(1, (training_data.timesteps - 1) * 5 + 1):
                model.update(i)
        except:
            costs.append(99999999)
            continue

        np_bg_model = np.array(pd.DataFrame(model.history)[g_label])
        np_bg_training = np.array(training_data.iob_data_frame)

        spline_factor = 0.01
        error = np.sum(np.square(np_bg_model - np_bg_training))
        error += np.sum(np.square(np.diff(np_bg_model))) * spline_factor

        if math.isinf(error) or math.isnan(error):
            costs.append(99999999)
            continue

        costs.append(error)

    return costs
from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.model import Model
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import os

def process(data_trace):
    step_dict = dict()

    for additional_timesteps in range(0, 25, 4):
        model_outputs = []
        fitnesses = []
        labels = []

        training_data = TrainingData(os.path.join("./data", data_trace))
        testing_data = TrainingData(os.path.join("./data", data_trace), 24 + additional_timesteps)
        np_bg_testing = np.array(testing_data.bg_data_frame)
        for _i in range(10):
            ga = GlucoseInsulinGeneticAlgorithm()
            constants, _f = ga.run(training_data)
            training_model = Model(training_data.find_initial_values(), constants)

            for i in range(1, (testing_data.timesteps - 1) * 5 + 1):
                training_model.update(i)

            np_bg_model = np.array(pd.DataFrame(training_model.history)[g_label])
            training_error = math.sqrt((1 / len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_testing)))

            fitnesses.append(training_error)
            labels.append(f"{data_trace[:-4]} {additional_timesteps}")

            model_outputs.append(np_bg_model)

        if data_trace[:-4] not in step_dict:
            step_dict[data_trace[:-4]] = []
        
        step_dict[data_trace[:-4]].append((len(np_bg_testing), np_bg_testing, model_outputs, fitnesses, labels))

    return step_dict

if __name__ == "__main__":
    figure_output = "<path>"

    all_traces = os.listdir("./data")

    all_fitnesses = []
    all_labels = []

    while len(all_traces) > 0:
        num = 50
        if num > len(all_traces):
            num = len(all_traces)

        with mp.Pool(processes=num) as pool:
            pool_vals = []
            for _i in range(num):
                data_trace = all_traces.pop()
                if data_trace.endswith(".csv"):
                    pool_vals.append(data_trace)
            
            resturnvals = pool.map(process, pool_vals)

            for step_dict in resturnvals:
                for input in step_dict.keys():
                    for steps, np_bg_testing, model_outputs, fitnesses, labels in step_dict[input]:
                        all_labels.extend(labels)
                        all_fitnesses.extend(fitnesses)

                        for model_output in model_outputs:
                            plt.plot(range(steps), model_output, c="black", linestyle="--", linewidth = 1, alpha = 0.1)

                        plt.plot(range(steps), np_bg_testing, c="red", linestyle="-", linewidth = 1, alpha = 1)
                        plt.ylim([0, 300])
                        plt.ylabel("Blood Glucose")
                        plt.xlabel("Timestep")
                        plt.title(f"Model output")
                        plt.savefig(f"{figure_output}/RQ2/{input}_{steps}")
                        plt.clf()

    data = {"Person": all_labels, "Error": all_fitnesses}
    df = pd.DataFrame(data=data)
    df.to_csv("rq2_errors.csv")
    ax = df.plot.scatter(x="Person", y="Error", c="black", s=4)
    ax.set_title("Error Across Model Training")
    plt.savefig(f"{figure_output}/RQ2/Errors")
from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData, i_label
from aps_digitaltwin.model import Model
from aps_digitaltwin.openaps import OpenAPS

import time
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from dotenv import load_dotenv

def process(data_trace):

    load_dotenv()

    training_data = TrainingData(os.path.join("./data", data_trace))
    ga = GlucoseInsulinGeneticAlgorithm()

    start = time.time()
    constants, _f = ga.run(training_data)
    contant_duration = time.time() - start
    
    training_model = Model(training_data.find_initial_values(), constants)
    
    start = time.time()
    for i in range(1, (training_data.timesteps - 1) * 10 + 1):
        training_model.update(i)
    model_duration = time.time() - start
    
    open_aps = OpenAPS(profile_path="./data/example_oref0_data/profiles/profile_100.json")
    open_aps_model = Model(training_data.find_initial_values(), constants)
    
    start = time.time()
    for i in range(1, (training_data.timesteps - 1) * 10 + 1):
        if i % 5 == 1:
            rate = open_aps.run(open_aps_model.history, f"./openaps_temp_{data_trace[:-4]}")
            for j in range(5):
                open_aps_model.add_intervention(i + j, i_label, rate / 5.0)
        open_aps_model.update(i)
    oref0_duration = time.time() - start
        
    return contant_duration, model_duration, oref0_duration

if __name__ == "__main__":
    figure_output = "<path>"
    
    all_traces = os.listdir("./data")

    all_constant_times = []
    all_model_times = []
    all_oref0_times = []
    
    num = 1
    with mp.Pool(processes=num) as pool:
        pool_vals = []
        for data_trace in all_traces:
            if data_trace.endswith(".csv"):
                pool_vals.append(data_trace)
        
        res_list = pool.map(process, pool_vals)

        for const, model, oref in res_list:
            all_constant_times.append(const)
            all_model_times.append(model)
            all_oref0_times.append(oref)

    data = pd.DataFrame([all_constant_times, all_model_times, all_oref0_times])
    data.to_csv("timings.csv")

    data = pd.read_csv("timings.csv").transpose()
    data.columns = ["Fitting Execution", "Model Execution", "oref0 Execution"]

    print(np.average(data["Fitting Execution"]))
    print(np.average(data["Model Execution"]))
    print(np.average(data["oref0 Execution"]))

    plt.boxplot(data, showfliers=False, labels=["Fitting", "Model Execution", "oref0 Execution"])
    plt.ylabel("time (s)")
    plt.title("Temporal Analysis")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_output, "RQ3.1.png"))
    plt.show()

from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData, g_label, i_label
from aps_digitaltwin.model import Model
from aps_digitaltwin.openaps import OpenAPS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from dotenv import load_dotenv

def process(data_trace):
    load_dotenv()
    model_outputs = {"none": [], "100": [], "120": [], "140": []}

    training_data = TrainingData(os.path.join("./data", data_trace))

    for _i in range(10):
        ga = GlucoseInsulinGeneticAlgorithm()
        constants, _f = ga.run(training_data)
        training_model = Model(training_data.find_initial_values(), constants)

        for i in range(1, (training_data.timesteps - 1) * 10 + 1):
            training_model.update(i)

        np_bg_model = np.array(pd.DataFrame(training_model.history)[g_label])

        model_outputs["none"].append(np_bg_model)

        for profile, name in [("./data/example_oref0_data/profiles/profile_140.json", "140"),
                                ("./data/example_oref0_data/profiles/profile_100.json", "100"),
                                ("./data/example_oref0_data/profiles/profile_120.json", "120")]:

            open_aps = OpenAPS(profile_path=profile)
            open_aps_model = Model(training_data.find_initial_values(), constants)

            for i in range(1, (training_data.timesteps - 1) * 10 + 1):
                if i % 5 == 1:
                    rate = open_aps.run(open_aps_model.history, f"./openaps_temp_{data_trace[:-4]}_{name}")
                    for j in range(5):
                        open_aps_model.add_intervention(i + j, i_label, rate / 5.0)
                open_aps_model.update(i)

            np_bg_model = np.array(pd.DataFrame(open_aps_model.history)[g_label])

            model_outputs[name].append(np_bg_model)
        
    return data_trace[:-4], model_outputs


if __name__ == "__main__":

    figure_output = "<path>"

    all_traces = os.listdir("./data")

    all_tir = []
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

            for data_trace, model_outputs in resturnvals:
                for outputs_none in model_outputs["none"]:
                    plt.plot(range(len(outputs_none)), outputs_none, c="black", linestyle="--", linewidth = 1, alpha = 0.5)
                    all_labels.append(f"{data_trace}_none")
                    all_tir.append(np.count_nonzero((outputs_none > 70) & (outputs_none < 180))/outputs_none.size)
                for outputs_140 in model_outputs["140"]:
                    plt.plot(range(len(outputs_140)), outputs_140, c="r", linestyle="--", linewidth = 1, alpha = 0.5)
                    all_labels.append(f"{data_trace}_140")
                    all_tir.append(np.count_nonzero((outputs_140 > 70) & (outputs_140 < 180))/outputs_140.size)
                for outputs_100 in model_outputs["100"]:
                    plt.plot(range(len(outputs_100)), outputs_100, c="b", linestyle="--", linewidth = 1, alpha = 0.5)
                    all_labels.append(f"{data_trace}_100")
                    all_tir.append(np.count_nonzero((outputs_100 > 70) & (outputs_100 < 180))/outputs_100.size)
                for outputs_120 in model_outputs["120"]:
                    plt.plot(range(len(outputs_120)), outputs_120, c="g", linestyle="--", linewidth = 1, alpha = 0.5)
                    all_labels.append(f"{data_trace}_120")
                    all_tir.append(np.count_nonzero((outputs_120 > 70) & (outputs_120 < 180))/outputs_120.size)
                plt.ylim([0, 300])
                plt.ylabel("Blood Glucose")
                plt.xlabel("Timestep")
                plt.title(f"OpenAPS Intervention")
                plt.savefig(f"{figure_output}/RQ3/{data_trace}")
                plt.clf()

    data = {"Person": all_labels, "TIR": all_tir}
    df = pd.DataFrame(data=data)
    df.to_csv("rq3_tir.csv")
    ax = df.plot.scatter(x="Person", y="TIR", c="black", s=4)
    ax.set_title("Error Across Model Training")
    plt.savefig(f"{figure_output}/RQ3/TIR")

from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.model import Model
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os

if __name__ == "__main__":
    fitnesses = []
    labels = []

    figure_save_path = "<path>"

    for data_trace in os.listdir("./data"):
        if data_trace.endswith(".csv"):
            training_data = TrainingData(os.path.join("./data", data_trace))
            np_bg_training = np.array(training_data.bg_data_frame)
            for _i in range(20):
                ga = GlucoseInsulinGeneticAlgorithm()
                constants, _f = ga.run(training_data)
                training_model = Model(training_data.find_initial_values(), constants)

                for intervention in training_data.interventions:
                    training_model.add_intervention(intervention[0], intervention[1], intervention[2])

                for i in range(1, (training_data.timesteps - 1) * 5 + 1):
                    training_model.update(i)

                np_bg_model = np.array(pd.DataFrame(training_model.history)[g_label])
                training_error = math.sqrt((1 / len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_training)))

                fitnesses.append(training_error)
                labels.append(f"Person {data_trace}")

                plt.plot(np.array(pd.DataFrame(training_model.history)["step"]), np_bg_model, 
                        c="black", linestyle="--", linewidth = 1, alpha = 0.1)
                
            plt.plot(range((training_data.timesteps - 1) * 5 + 1), np_bg_training, 
                        c="red", linestyle="-", linewidth = 1, alpha = 1)
            plt.ylim([0, 200])
            plt.ylabel("Blood Glucose")
            plt.xlabel("Timestep")
            plt.title(f"Model output")
            plt.savefig(f"{figure_save_path}/RQ1/Person_{data_trace}")
            plt.clf()
        
    data = {"Person": labels, "Error": fitnesses}
    df = pd.DataFrame(data=data)
    df.to_csv("rq1_errors.csv")
    ax = df.plot.scatter(x="Person", y="Error", c="black", s=4)
    ax.set_title("Error Across Model Training")
    ax.set_yscale("log")
    plt.savefig(f"{figure_save_path}/RQ1/Errors")
from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.model import Model
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fitnesses = []
    labels = []

    figure_save_path = "<path>"

    for person in range(1,4):
        for additional_timesteps in range(0, 25, 4):
            training_data = TrainingData(f"./data/data_{person}.csv")
            testing_data = TrainingData(f"./data/data_{person}.csv", 24 + additional_timesteps)
            np_bg_testing = np.array(testing_data.bg_data_frame)
            for _i in range(20):
                ga = GlucoseInsulinGeneticAlgorithm()
                constants, _f = ga.run(training_data)
                training_model = Model(training_data.find_initial_values(), constants)

                for i in range(1, (testing_data.timesteps - 1) * 5 + 1):
                    training_model.update(i)

                np_bg_model = np.array(pd.DataFrame(training_model.history)[g_label])
                training_error = math.sqrt((1 / len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_testing)))

                fitnesses.append(training_error)
                labels.append(f"Person {person} {additional_timesteps}")

                plt.plot(np.array(pd.DataFrame(training_model.history)["step"]), np_bg_model, 
                        c="black", linestyle="--", linewidth = 1, alpha = 0.1)
                
            plt.plot(range((testing_data.timesteps - 1) * 5 + 1), np_bg_testing, 
                        c="red", linestyle="-", linewidth = 1, alpha = 1)
            plt.ylim([0, 200])
            plt.ylabel("Blood Glucose")
            plt.xlabel("Timestep")
            plt.title(f"Model output interpolating {additional_timesteps} timesteps")
            plt.savefig(f"{figure_save_path}/RQ2/Person_{person}_{additional_timesteps}")
            plt.clf()

        data = {"Person": labels, "Error": fitnesses}
        df = pd.DataFrame(data=data)
        df.to_csv("rq2_errors.csv")

            
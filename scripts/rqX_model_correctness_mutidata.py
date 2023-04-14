from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.model import Model
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fitnesses = []
    labels = []

    figure_save_path = "<path>"

    for person in range(1,4):
        training_data = [TrainingData(f"./data/data_{person}.csv"),
                         TrainingData(f"./data/data_{person}_test.csv")]

        saved_traces = dict()
        for idx, _data in enumerate(training_data):
            saved_traces[idx] = []

        for _i in range(20):
            ga = GlucoseInsulinGeneticAlgorithm()
            constants, _f = ga.run(training_data)

            for idx, dataset in enumerate(training_data):
                training_model = Model(dataset.find_initial_values(), constants)

                for intervention in dataset.interventions:
                    training_model.add_intervention(intervention[0], intervention[1], intervention[2])

                for i in range(1, (dataset.timesteps - 1) * 5 + 1):
                    training_model.update(i)

                np_bg_model = np.array(pd.DataFrame(training_model.history)[g_label])

                saved_traces[idx].append(np_bg_model)

        for (key, dataset) in saved_traces.items():
            for trace in dataset:
                plt.plot(np.array(pd.DataFrame(training_model.history)["step"]), trace, 
                        c="black", linestyle="--", linewidth = 1, alpha = 0.1)
            
            np_bg_training = np.array(training_data[key].bg_data_frame)
            plt.plot(range((training_data[0].timesteps - 1) * 5 + 1), np_bg_training, 
                        c="red", linestyle="-", linewidth = 1, alpha = 1)
            plt.ylim([0, 200])
            plt.ylabel("Blood Glucose")
            plt.xlabel("Timestep")
            plt.title(f"Person {person} Model Outputs {key + 1}")
            plt.savefig(f"{figure_save_path}/RQ2/Person_{person}_{key + 1}")
            plt.clf()
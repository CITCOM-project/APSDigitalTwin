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

    figure_save_path = "/home/richardsomers/Desktop"

    # for person in range(1,4):
    for person in [2]:
        training_data = TrainingData(f"./data/data_{person}.csv")
        testing_data = TrainingData(f"./data/data_{person}_test.csv")
        np_bg_testing = np.array(testing_data.bg_data_frame)
        for _i in range(20):
            ga = GlucoseInsulinGeneticAlgorithm()
            constants, _f = ga.run(training_data)

            testing_model = Model(testing_data.find_initial_values(), constants)

            for intervention in testing_data.interventions:
                testing_model.add_intervention(intervention[0], intervention[1], intervention[2])

            for i in range(1, (testing_data.timesteps - 1) * 5 + 1):
                testing_model.update(i)

            np_bg_model = np.array(pd.DataFrame(testing_model.history)[g_label])
            testing_error = np.sum(np.square(np_bg_model - np_bg_testing))
            testing_fitness = 1/testing_error

            fitnesses.append(testing_fitness)
            labels.append(f"Person {person}")

            plt.plot(np.array(pd.DataFrame(testing_model.history)["step"]), np_bg_model, 
                     c="black", linestyle="--", linewidth = 1, alpha = 0.1)

        plt.plot(range((training_data.timesteps - 1) * 5 + 1), np_bg_testing, 
                     c="red", linestyle="-", linewidth = 1, alpha = 1)
        plt.ylim([0, 200])
        plt.ylabel("Blood Glucose")
        plt.xlabel("Timestep")
        plt.title(f"Person {person} Interpolation Outputs")
        plt.savefig(f"{figure_save_path}/RQ2/Person_{person}")
        plt.clf()

    # data = {"Person": labels, "Fitness": fitnesses}

    # df = pd.DataFrame(data=data)
    # df.to_csv("rq2_fitnesses.csv")
    # ax = df.plot.scatter(x="Person", y="Fitness", c="black", s=4)
    # ax.set_title("Fitness Across Model Interpolation")
    # ax.set_yscale("log")
    # plt.savefig(f"{figure_save_path}/RQ2/Fitnesses")
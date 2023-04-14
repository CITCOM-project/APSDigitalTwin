from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData, g_label, i_label
from aps_digitaltwin.model import Model
from aps_digitaltwin.openaps import OpenAPS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()
    plt.clf()

    figure_save_path = "<path>"

    for person in range(1,4):
        training_data = [TrainingData(f"./data/data_{person}.csv")]

        ga = GlucoseInsulinGeneticAlgorithm()
        constants, _f = ga.run(training_data)
        training_model = Model(training_data[0].find_initial_values(), constants)

        for i in range(1, (training_data[0].timesteps - 1) * 10 + 1):
            training_model.update(i)

        np_bg_model = np.array(pd.DataFrame(training_model.history)[g_label])

        plt.plot(np.array(pd.DataFrame(training_model.history)["step"]), np_bg_model, 
                    c="black", linestyle="--", linewidth = 1, alpha = 1)

        for profile, colour in [("./data/example_oref0_data/profiles/profile_80.json", "r"),
                                ("./data/example_oref0_data/profiles/profile_100.json", "b"),
                                ("./data/example_oref0_data/profiles/profile_120.json", "g")]:

            open_aps = OpenAPS(profile_path=profile)
            open_aps_model = Model(training_data[0].find_initial_values(), constants)

            for i in range(1, (training_data[0].timesteps - 1) * 10 + 1):
                if i % 5 == 1:
                    rate = open_aps.run(open_aps_model.history)
                    for j in range(5):
                        open_aps_model.add_intervention(i + j, i_label, rate / 5.0)
                open_aps_model.update(i)

            np_bg_model = np.array(pd.DataFrame(open_aps_model.history)[g_label])

            plt.plot(np.array(pd.DataFrame(open_aps_model.history)["step"]), np_bg_model, 
                    c=colour, linestyle="--", linewidth = 1, alpha = 1)
            
        plt.ylim([0, 200])
        plt.ylabel("Blood Glucose")
        plt.xlabel("Timestep")
        plt.title(f"Person {person} OpenAPS Intervention")
        plt.savefig(f"{figure_save_path}/RQ3/Person_{person}")
        plt.clf()
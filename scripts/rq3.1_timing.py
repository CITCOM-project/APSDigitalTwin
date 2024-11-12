from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData, i_label
from aps_digitaltwin.model import Model
from aps_digitaltwin.openaps import OpenAPS

import time
import numpy as np

from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    training_data = TrainingData("./data/data_1.csv")
    ga = GlucoseInsulinGeneticAlgorithm()
    
    constant_times = []
    model_times = []
    oref0_times = []
    
    for _i in range(10):
        start = time.time()
        constants, _f = ga.run(training_data)
        duration = time.time() - start
        constant_times.append(duration)
        
        training_model = Model(training_data.find_initial_values(), constants)
        
        start = time.time()
        for i in range(1, (training_data.timesteps - 1) * 10 + 1):
            training_model.update(i)
        duration = time.time() - start
        model_times.append(duration)
        
        open_aps = OpenAPS(profile_path="./data/example_oref0_data/profiles/profile_100.json")
        open_aps_model = Model(training_data.find_initial_values(), constants)
        
        start = time.time()
        for i in range(1, (training_data.timesteps - 1) * 10 + 1):
            if i % 5 == 1:
                rate = open_aps.run(open_aps_model.history)
                for j in range(5):
                    open_aps_model.add_intervention(i + j, i_label, rate / 5.0)
            open_aps_model.update(i)
        duration = time.time() - start
        oref0_times.append(duration)
        
    with open("timings.txt", "w") as file:
        file.write(f"Constants: {np.mean(constant_times)}\n")
        file.write(f"Model: {np.mean(model_times)}\n")
        file.write(f"oref0: {np.mean(oref0_times)}\n")
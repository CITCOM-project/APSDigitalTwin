from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario
from aps_digitaltwin.model import Model
import numpy as np

if __name__ == "__main__":

    profile = "./data/example_oref0_data/profile.json"
    basal_profile = "./data/example_oref0_data/basal_profile.json"

    training_data = TrainingData("./data/data.csv")

    ga = GlucoseInsulinGeneticAlgorithm()
    constants = ga.run(training_data)

    # constants = [0.006087855309129808, 0.8169421695707391, 0.2341139159982908, 0.7514639148751912, 0.003159062595706197, 0.00033439542508328923, 0.026713764597784406, 124, 0.21414131324524588, 0.993974927000483, 0.007151843859299123, 0.7580049521993947]

    s1 = Scenario(30, 70, 20, 120, 180, 40, []) # Low Crashing
    s2 = Scenario(50, 70, 0, 120, 180, 40, []) # Low Stable
    s3 = Scenario(200, 70, 0, 120, 180, 40, []) # Low Rising

    s4 = Scenario(30, 100, 20, 120, 180, 40, []) # Normal Crashing
    s5 = Scenario(50, 100, 0, 120, 180, 40, []) # Normal Stable
    s6 = Scenario(200, 100, 0, 120, 180, 40, []) # Normal Rising

    s7 = Scenario(30, 130, 20, 120, 180, 40, []) # High Crashing
    s8 = Scenario(50, 130, 0, 120, 180, 40, []) # High Stable
    s9 = Scenario(200, 130, 0, 120, 180, 40, []) # High Rising

    print(s1.run(constants, profile, basal_profile))
    print(s2.run(constants, profile, basal_profile))
    print(s3.run(constants, profile, basal_profile))
    print(s4.run(constants, profile, basal_profile))
    print(s5.run(constants, profile, basal_profile))
    print(s6.run(constants, profile, basal_profile))
    print(s7.run(constants, profile, basal_profile))
    print(s8.run(constants, profile, basal_profile))
    print(s9.run(constants, profile, basal_profile))

    # for x in range(400):
    #     carbs = 70 + np.random.normal() * 50
    #     blood_glucose = 100 + np.random.normal() * 30
    #     recorded_carbs = carbs + (carbs * 0.2 * np.random.normal())
    #     insulin = 20 + np.random.normal() * 10 if np.random.normal() > 0.0 else 0

    #     scenario = Scenario(carbs, blood_glucose, insulin, 120, 180, 40, [])
    #     scenario.run(constants, profile, basal_profile, recorded_carbs, "./output_3.csv")

    # constants = [0.006130878853835453, 0.46253313443648725, 0.905360146129958, 0.907250092652974, 0.0014680021770052676, 0.0001753511527067264, 0.05578541545495119, 118, 0.3353304947598008, 0.11723477928743364, 0.02342579713860593]

    # model = Model(training_data.find_initial_values(), constants)

    # for intervention in training_data.interventions:
    #     model.add_intervention(intervention[0], intervention[1], intervention[2])

    # try:
    #     for i in range(1, (training_data.timesteps - 1) * 5 + 1):
    #         model.update(i)
    # except:
    #     raise Exception("Model learning failed")
    
    # model.plot()
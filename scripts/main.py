from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario
from aps_digitaltwin.model import Model
import numpy as np

from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    training_data_1 = TrainingData("./data/data.csv")
    ga_1 = GlucoseInsulinGeneticAlgorithm()
    constants_1 = ga_1.run(training_data_1)

    training_data_2 = TrainingData("./data/data_2.csv")
    ga_2 = GlucoseInsulinGeneticAlgorithm()
    constants_2 = ga_2.run(training_data_2)

    training_data_2 = TrainingData("./data/data_3.csv")
    ga_2 = GlucoseInsulinGeneticAlgorithm()
    constants_3 = ga_2.run(training_data_2)

    # constants = [0.006466690185883084, 0.9986775362706466, 0.35617346722058174, 0.23659927959841165, 0.003982946883520522, 3.9495416978785336e-05, 0.027624490511824584, 115, 1.0357696550666873e-05, 0.5536124833434388, 0.009252262930285915, 0.9768732305922168]

    # s1 = Scenario(30, 70, 20, 120, 180, 40, []) # Low Crashing
    # s2 = Scenario(50, 70, 0, 120, 180, 40, []) # Low Stable
    # s3 = Scenario(200, 70, 0, 120, 180, 40, []) # Low Rising

    # s4 = Scenario(30, 100, 20, 120, 180, 40, []) # Normal Crashing
    # s5 = Scenario(50, 100, 0, 120, 180, 40, []) # Normal Stable
    # s6 = Scenario(200, 100, 0, 120, 180, 40, []) # Normal Rising

    # s7 = Scenario(30, 130, 20, 120, 180, 40, []) # High Crashing
    # s8 = Scenario(50, 130, 0, 120, 180, 40, []) # High Stable
    # s9 = Scenario(200, 130, 0, 120, 180, 40, []) # High Rising

    # print(s1.run(constants))
    # print(s2.run(constants))
    # print(s3.run(constants))
    # print(s4.run(constants))
    # print(s5.run(constants))
    # print(s6.run(constants))
    # print(s7.run(constants))
    # print(s8.run(constants))
    # print(s9.run(constants))

    for x in range(400):
        carbs = 70 + np.random.normal() * 50
        blood_glucose = 100 + np.random.normal() * 30
        recorded_carbs = carbs + (carbs * 0.2 * np.random.normal())
        insulin = 20 + np.random.normal() * 10 if np.random.normal() > 0.0 else 0

        scenario = Scenario(carbs, blood_glucose, insulin, 120, 180, 40, [])
        scenario.run(constants_1, recorded_carbs, "./output_1.csv")

    for x in range(400):
        carbs = 70 + np.random.normal() * 50
        blood_glucose = 100 + np.random.normal() * 30
        recorded_carbs = carbs + (carbs * 0.2 * np.random.normal())
        insulin = 20 + np.random.normal() * 10 if np.random.normal() > 0.0 else 0

        scenario = Scenario(carbs, blood_glucose, insulin, 120, 180, 40, [])
        scenario.run(constants_2, recorded_carbs, "./output_2.csv")
        
    for x in range(400):
        carbs = 70 + np.random.normal() * 50
        blood_glucose = 100 + np.random.normal() * 30
        recorded_carbs = carbs + (carbs * 0.2 * np.random.normal())
        insulin = 20 + np.random.normal() * 10 if np.random.normal() > 0.0 else 0

        scenario = Scenario(carbs, blood_glucose, insulin, 120, 180, 40, [])
        scenario.run(constants_3, recorded_carbs, "./output_3.csv")

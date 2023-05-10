from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario

from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    training_data = TrainingData("./data/data_1.csv")
    ga = GlucoseInsulinGeneticAlgorithm()
    constants, _f = ga.run(training_data, True)

    # constants = [0.006099608383310695, 0.9480043796363652, 0.3658330385151397, 0.09029516786725544, 0.0016741618977367256, 0.00026132989388338856, 0.043580989667546755, 114, 0.9727020894984222, 0.780562469959468, 0.9902416909564876, 0.1588669031447998]

    s1 = Scenario(30, 70, 20, 120, 180, 40, []) # Low Crashing
    s2 = Scenario(50, 70, 0, 120, 180, 40, []) # Low Stable
    s3 = Scenario(200, 70, 0, 120, 180, 40, []) # Low Rising

    s4 = Scenario(30, 100, 20, 120, 180, 40, []) # Normal Crashing
    s5 = Scenario(50, 100, 0, 120, 180, 40, []) # Normal Stable
    s6 = Scenario(200, 100, 0, 120, 180, 40, []) # Normal Rising

    s7 = Scenario(30, 130, 20, 120, 180, 40, []) # High Crashing
    s8 = Scenario(50, 130, 0, 120, 180, 40, []) # High Stable
    s9 = Scenario(200, 130, 0, 120, 180, 40, []) # High Rising

    print(s1.run(constants))
    print(s2.run(constants))
    print(s3.run(constants))
    print(s4.run(constants))
    print(s5.run(constants))
    print(s6.run(constants))
    print(s7.run(constants))
    print(s8.run(constants))
    print(s9.run(constants))
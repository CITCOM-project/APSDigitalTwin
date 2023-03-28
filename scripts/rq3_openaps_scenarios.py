from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario

from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    training_data_1 = TrainingData("./data/data_1.csv")
    ga_1 = GlucoseInsulinGeneticAlgorithm()
    constants, _f = ga_1.run(training_data_1)

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
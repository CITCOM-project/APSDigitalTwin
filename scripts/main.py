from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario

from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    training_data = TrainingData("./data/data.csv")

    ga = GlucoseInsulinGeneticAlgorithm()
    constants = ga.run(training_data)

    # constants = [0.006087855309129808, 0.8169421695707391, 0.2341139159982908, 0.7514639148751912, 0.003159062595706197, 0.00033439542508328923, 0.026713764597784406, 124, 0.21414131324524588, 0.993974927000483, 0.007151843859299123, 0.7580049521993947]

    s1 = Scenario(30, 70, 50, 120, 180, 40, []) # Low Crashing
    s2 = Scenario(50, 70, 0, 120, 180, 40, []) # Low Stable
    s3 = Scenario(200, 70, 0, 120, 180, 40, []) # Low Rising

    s4 = Scenario(30, 100, 50, 120, 180, 40, []) # Normal Crashing
    s5 = Scenario(50, 100, 0, 120, 180, 40, []) # Normal Stable
    s6 = Scenario(200, 100, 0, 120, 180, 40, []) # Normal Rising

    s7 = Scenario(30, 130, 50, 120, 180, 40, []) # High Crashing
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
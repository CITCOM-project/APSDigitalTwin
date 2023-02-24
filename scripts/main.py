from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario

if __name__ == "__main__":

    profile = "./data/example_oref0_data/profile.json"
    basal_profile = "./data/example_oref0_data/basal_profile.json"

    training_data = TrainingData("./data/data.csv")

    ga = GlucoseInsulinGeneticAlgorithm()
    constants = ga.run(training_data)

    # constants = [0.006237490718434935, 0.0016178369029404838, 0.4591610516833179, 0.5753904673542842, 0.00035958241954792136, 0.0011730992937690754, 0.019451673788506763, 38, 0.6035644640425673, 0.9802986866743725, 0.0022734639570410886, 0.8963363254601819]

    s1 = Scenario(30, 70, 50, 120, 180, 40, []) # Low Crashing
    s2 = Scenario(50, 70, 0, 120, 180, 40, []) # Low Stable
    s3 = Scenario(200, 70, 0, 120, 180, 40, []) # Low Rising

    s4 = Scenario(30, 100, 50, 120, 180, 40, []) # Normal Crashing
    s5 = Scenario(50, 100, 0, 120, 180, 40, []) # Normal Stable
    s6 = Scenario(200, 100, 0, 120, 180, 40, []) # Normal Rising

    s7 = Scenario(30, 130, 50, 120, 180, 40, []) # High Crashing
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
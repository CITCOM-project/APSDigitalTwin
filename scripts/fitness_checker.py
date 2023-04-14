from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.swarm import GlucoseInsulinParticleSwarm
from aps_digitaltwin.util import TrainingData

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    training_data = [TrainingData("./data/data_1.csv")]
    df = pd.DataFrame()

    iterations = 20
    labels = []
    fitnesses = []

    for x in range(iterations):
        ps = GlucoseInsulinParticleSwarm(swarm=False)
        constants, fitness = ps.run(training_data)
        labels.append("Gradient Search")
        fitnesses.append(fitness)

    for x in range(iterations):
        ps = GlucoseInsulinParticleSwarm()
        constants, fitness = ps.run(training_data)
        labels.append("Particle Swarm")
        fitnesses.append(fitness)

    for x in range(iterations):
        ga = GlucoseInsulinGeneticAlgorithm()
        constants, fitness = ga.run(training_data)
        labels.append("Genetic Algorithm")
        fitnesses.append(1/fitness)

    data = {"Algorithm": labels, "Fitness": fitnesses}

    df = pd.DataFrame(data=data)
    ax = df.plot.scatter(x="Algorithm", y="Fitness", c="black", s=4)
    ax.set_title("Fitness Across Fitting Algorithms")
    ax.set_yscale("log")
    ax.set_ylim(top=0, bottom=10e-13)
    plt.show()
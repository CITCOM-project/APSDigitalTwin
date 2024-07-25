from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm
from aps_digitaltwin.swarm import GlucoseInsulinParticleSwarm
from aps_digitaltwin.annealing import GlucoseInsulinAnnealing
from aps_digitaltwin.util import TrainingData

import os
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import random
import numpy as np

if __name__ == "__main__":

    random.seed(1234)
    np.random.seed(1234)

    iterations = 10
    labels = []
    fitnesses = []

    sampled_traces = random.sample([x for x in os.listdir("./data") if x.endswith(".csv")], 10)

    for trace in sampled_traces:
        training_data = TrainingData(f"./data/{trace}")
        df = pd.DataFrame()
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
            fitnesses.append(fitness)
        
        for x in range(iterations):
            ga = GlucoseInsulinAnnealing()
            constants, fitness = ga.run(training_data)
            labels.append("Simulated Annealing")
            fitnesses.append(fitness)

    data = {"Algorithm": labels, "Fitness": fitnesses}

    df = pd.DataFrame(data=data)
    df.to_csv("sampled_fitnesses.csv")

    # df = pd.read_csv("sampled_fitnesses.csv")

    sns.set_style("whitegrid")
    sns.boxplot(data=df, x="Algorithm", y="Fitness", log_scale=True, palette="colorblind")
    ax = plt.gca()
    ax.set_title("Fitness Across Fitting Algorithms")
    plt.ylim(10e-5, 1)
    plt.show()
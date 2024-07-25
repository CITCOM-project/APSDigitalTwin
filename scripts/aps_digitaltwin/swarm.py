import pyswarms as ps
import numpy as np
from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.fitness_functions import fitness_function_stomach, fitness_function_insulin, fitness_function_glucose

class GlucoseInsulinParticleSwarm:

    def __init__(self, swarm = True) -> None:
        self.__kjs = 0
        self.__kxi = 0
        self.swarm = swarm

        self.training_data = None

    def run(self, training_data):
        self.training_data = training_data

        options = {'c1': 0.5, 'c2': 0.5, 'w':0.8}
        optimiser = ps.single.GlobalBestPSO(n_particles=10 if self.swarm else 1, dimensions=1, options=options, bounds=([0],[1]))
        cost, pos = optimiser.optimize(fitness_function_stomach, iters=100, training_data=training_data)
        self.__kjs = pos[0]

        optimiser2 = ps.single.GlobalBestPSO(n_particles=20 if self.swarm else 1, dimensions=1, options=options, bounds=([0],[1]))
        cost, pos2 = optimiser2.optimize(fitness_function_insulin, iters=100, training_data=training_data, kjs=self.__kjs)
        self.__kxi = pos2[0]

        bounds = ([0,0,0,0,0,1,0,0,0,0],
                  [1,1,1,1,1,self.training_data.timesteps * 5, 1,1,1,10])

        optimiser3 = ps.single.GlobalBestPSO(n_particles=100 if self.swarm else 1, dimensions=10, options=options, bounds=bounds)
        cost, pos3 = optimiser3.optimize(fitness_function_glucose, iters=100, training_data=training_data,
                                         kjs=self.__kjs, kxi=self.__kxi)
        
        best_constants = [
            pos[0],
            pos3[0],
            pos3[1],
            pos3[2],
            pos3[3],
            pos3[4],
            pos2[0],
            round(pos3[5]),
            pos3[6],
            pos3[7],
            pos3[8],
            pos3[9]
        ]

        return best_constants, 1/cost
    
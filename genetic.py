import pygad
import math
import pandas as pd
import numpy as np
from model import Model
from setup import s_label, g_label, i_label

class GlucoseInsulinGeneticAlgorithm:

    def __init__(self) -> None:
        self.__kjs = 0
        self.__kxi = 0
        self.__ib = 0

        self.training_data = None

    def run(self, training_data):
        self.training_data = training_data

        ga_stomach_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=3,
                       fitness_func=self.wrapped_fitness_function_stomach(),
                       sol_per_pop=10,
                       num_genes=1,
                       mutation_type="random",
                       mutation_percent_genes=100,
                       gene_space=[{"low": 0, "high": 1}],
                       gene_type=[np.float64])

        ga_stomach_instance.run()

        stomach_solution, stomach_solution_fitness, solution_idx = ga_stomach_instance.best_solution()
        print(f"Fitness value of kjs : {stomach_solution_fitness}")

        self.__kjs = stomach_solution[0]

        ga_insulin_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=3,
                       fitness_func=self.wrapped_fitness_function_insulin(),
                       sol_per_pop=10,
                       num_genes=2,
                       mutation_type="random",
                       mutation_percent_genes=80,
                       gene_space=[
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1}
                       ],
                       gene_type=[
                        np.float64,
                        np.float64
                       ])

        ga_stomach_instance.run()

        insulin_solution, insulin_solution_fitness, solution_idx = ga_insulin_instance.best_solution()
        print(f"Fitness value of kxi and Ib : {insulin_solution_fitness}")

        self.__kxi = insulin_solution[0]
        self.__ib = insulin_solution[1]

        ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=5,
                       fitness_func=self.wrapped_fitness_function_glucose(),
                       sol_per_pop=20,
                       num_genes=9,
                       mutation_type="random",
                       mutation_percent_genes=20,
                       gene_space=[
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 1, "high": self.training_data.timesteps * 5},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1}
                       ],
                       gene_type=[
                        np.float64,
                        np.float64,
                        np.float64,
                        np.float64,
                        np.float64,
                        int,
                        np.float64,
                        np.float64,
                        np.float64
                       ])

        ga_instance.run()

        final_solution, final_solution_fitness, final_solution_idx = ga_instance.best_solution()
        print(f"Fitness value of the final solution : {final_solution_fitness}")

        best_constants = [
            stomach_solution[0],
            final_solution[0],
            final_solution[1],
            final_solution[2],
            final_solution[3],
            final_solution[4],
            insulin_solution[0],
            final_solution[5],
            final_solution[6],
            final_solution[7],
            final_solution[8],
            insulin_solution[1]
        ]

        print(f"Best constants = [{', '.join(map(str, best_constants))}]")

        return best_constants

    def wrapped_fitness_function_stomach(self):

        def fitness_function_stomach(solution, solution_idx):
            constants = [solution[0],0.1,0.1,0.1,0.1,0.1,0.1,5,0.1,0.1,0.1,0.1]

            model = Model(self.training_data.find_initial_values(), constants)

            for intervention in self.training_data.interventions:
                model.add_intervention(intervention[0], intervention[1], intervention[2])

            try:
                for i in range(1, (self.training_data.timesteps - 1) * 5 + 1):
                    model.update(i)
            except:
                return 0
            
            stomach_model_df = pd.DataFrame(model.history, dtype=np.float64)[s_label]

            error = 0
            for i in range(len(self.training_data.cob_data_frame)):
                diff = stomach_model_df[i] - self.training_data.cob_data_frame[i]
                diff_sq = diff ** 2
                error += diff_sq

            return 1/error
        
        return fitness_function_stomach
    
    def wrapped_fitness_function_insulin(self):

        def fitness_function_insulin(solution, solution_idx):
            constants = [self.__kjs,0.1,0.1,0.1,0.1,0.1,solution[0],5,0.1,0.1,0.1,solution[1]]

            model = Model(self.training_data.find_initial_values(), constants)

            for intervention in self.training_data.interventions:
                model.add_intervention(intervention[0], intervention[1], intervention[2])

            try:
                for i in range(1, (self.training_data.timesteps - 1) * 5 + 1):
                    model.update(i)
            except:
                return 0
            
            insulin_model_df = pd.DataFrame(model.history, dtype=np.float64)[i_label]

            spline_factor = 0.01
            
            error = 0
            for i in range(len(self.training_data.iob_data_frame)):
                diff = insulin_model_df[i] - self.training_data.iob_data_frame[i]
                diff_sq = diff ** 2
                error += diff_sq

            for i in range(len(insulin_model_df) - 1):
                diff = insulin_model_df[i] - insulin_model_df[i + 1]
                diff_sq = diff ** 2
                error += diff_sq * spline_factor

            return 1/error
        
        return fitness_function_insulin
    
    def wrapped_fitness_function_glucose(self):

        def fitness_function_glucose(solution, solution_idx):
            constants = [
                self.__kjs,
                solution[0],
                solution[1],
                solution[2],
                solution[3],
                solution[4],
                self.__kxi,
                solution[5],
                solution[6],
                solution[7],
                solution[8],
                self.__ib
            ]

            model = Model(self.training_data.find_initial_values(), constants)

            for intervention in self.training_data.interventions:
                model.add_intervention(intervention[0], intervention[1], intervention[2])

            try:
                for i in range(1, (self.training_data.timesteps - 1) * 5 + 1):
                    model.update(i)
            except:
                return 0

            bg_model_df = pd.DataFrame(model.history, dtype=np.float64)[g_label]

            spline_factor = 0.01

            error = 0
            for i in range(len(self.training_data.bg_data_frame)):
                diff = bg_model_df[i] - self.training_data.bg_data_frame[i]
                diff_sq = diff ** 2
                error += diff_sq

            for i in range(len(bg_model_df) - 1):
                diff = bg_model_df[i] - bg_model_df[i + 1]
                diff_sq = diff ** 2
                error += diff_sq * spline_factor

            if math.isinf(error) or math.isnan(error):
                return 0

            return 1/error
        
        return fitness_function_glucose
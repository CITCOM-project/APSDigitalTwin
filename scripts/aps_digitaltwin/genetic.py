import pygad
import math
import pandas as pd
import numpy as np
import warnings
from aps_digitaltwin.model import Model
from aps_digitaltwin.util import s_label, g_label, i_label

class GlucoseInsulinGeneticAlgorithm:

    def __init__(self) -> None:
        self.__kjs = 0
        self.__kxi = 0

        self.training_data = None

    def run(self, training_data, plot_model = False):
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
                       num_genes=1,
                       mutation_type="random",
                       mutation_percent_genes=100,
                       gene_space=[
                        {"low": 0, "high": 1}
                       ],
                       gene_type=[
                        np.float64
                       ])

        ga_stomach_instance.run()

        insulin_solution, insulin_solution_fitness, solution_idx = ga_insulin_instance.best_solution()
        print(f"Fitness value of kxi : {insulin_solution_fitness}")

        self.__kxi = insulin_solution[0]

        ga_instance = pygad.GA(num_generations=1500,
                       num_genes=10,
                       sol_per_pop=30,
                       fitness_func=self.wrapped_fitness_function_glucose(),
                       num_parents_mating=4,
                       parent_selection_type="tournament",
                       K_tournament=4,
                       mutation_type="random",
                       mutation_percent_genes=20,
                       mutation_by_replacement=True,
                       gene_space=[
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 1, "high": self.training_data.timesteps * 5},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 1},
                        {"low": 0, "high": 10}
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
            final_solution[9]
        ]

        print(f"Best constants = [{', '.join(map(str, best_constants))}]")

        if plot_model:
            model = Model(self.training_data.find_initial_values(), best_constants)

            for intervention in self.training_data.interventions:
                model.add_intervention(intervention[0], intervention[1], intervention[2])

            try:
                for i in range(1, (self.training_data.timesteps - 1) * 5 + 1):
                    model.update(i)
            except:
                raise Exception("Model learning failed")
            
            model.plot()

        return best_constants, final_solution_fitness

    def wrapped_fitness_function_stomach(self):

        def fitness_function_stomach(solution, solution_idx):
            constants = [solution[0],0.1,0.1,0.1,0.1,0.1,0.1,5,0.1,0.1,0.1,0.1]
            error = 0

            model = Model(self.training_data.find_initial_values(), constants)

            for intervention in self.training_data.interventions:
                model.add_intervention(intervention[0], intervention[1], intervention[2])

            try:
                for i in range(1, (self.training_data.timesteps - 1) * 5 + 1):
                    model.update(i)
            except:
                return 0
            
            np_stomach_model = np.array(pd.DataFrame(model.history)[s_label])
            np_stomach_training = np.array(self.training_data.cob_data_frame)
            error += np.sum(np.square(np_stomach_model - np_stomach_training))

            return 1/error
        
        return fitness_function_stomach
    
    def wrapped_fitness_function_insulin(self):

        def fitness_function_insulin(solution, solution_idx):
            constants = [self.__kjs,0.1,0.1,0.1,0.1,0.1,solution[0],5,0.1,0.1,0.1,0.1]
            error = 0

            model = Model(self.training_data.find_initial_values(), constants)

            for intervention in self.training_data.interventions:
                model.add_intervention(intervention[0], intervention[1], intervention[2])

            try:
                for i in range(1, (self.training_data.timesteps - 1) * 5 + 1):
                    model.update(i)
            except:
                return 0
            
            np_insulin_model = np.array(pd.DataFrame(model.history)[i_label])
            np_insulin_training = np.array(self.training_data.iob_data_frame)
            error += np.sum(np.square(np_insulin_model - np_insulin_training))

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
                solution[9]
            ]
            error = 0
            
            model = Model(self.training_data.find_initial_values(), constants)

            for intervention in self.training_data.interventions:
                model.add_intervention(intervention[0], intervention[1], intervention[2])

            try:
                for i in range(1, (self.training_data.timesteps - 1) * 5 + 1):
                    model.update(i)
            except:
                return 0

            np_bg_model = np.array(pd.DataFrame(model.history)[g_label])
            np_bg_training = np.array(self.training_data.bg_data_frame)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                spline_factor = 0.1
                error += math.sqrt((1/len(np_bg_model)) * np.sum(np.square(np_bg_model - np_bg_training)))
                error += math.sqrt((1/len(np_bg_model)) * np.sum(np.square(np.diff(np_bg_model))) * spline_factor)

            if math.isinf(error) or math.isnan(error):
                return 0

            return 1/error
        
        return fitness_function_glucose
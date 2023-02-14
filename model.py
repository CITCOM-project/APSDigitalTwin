from setup import s_label, j_label, l_label, g_label, i_label
import pandas as pd
import matplotlib.pyplot as plt

class Model:
    def __init__(self, starting_vals, constants):
        self.interventions = dict()

        self.history = []
        self.history.append({'step': 0, 
                             s_label: starting_vals[0], 
                             j_label: starting_vals[1], 
                             l_label: starting_vals[2], 
                             g_label: starting_vals[3], 
                             i_label: starting_vals[4]})

        self.kjs = constants[0]
        self.kgj = constants[1]
        self.kjl = constants[2]
        self.kgl = constants[3]
        self.kxg = constants[4]
        self.kxgi = constants[5]
        self.kxi = constants[6]

        self.tau = constants[7]
        self.klambda = constants[8]
        self.mu = constants[9]
        # self.beta = constants[10]
        # self.gamma = constants[11]

        self.gprod0 = constants[10]
        self.ib = constants[11]

        self.gb = starting_vals[0]

        # self.fgj = constants[15]


    def update(self, t):
        old_s = self.history[t-1][s_label]
        old_j = self.history[t-1][j_label]
        old_l = self.history[t-1][l_label]
        old_g = self.history[t-1][g_label]
        old_i = self.history[t-1][i_label]

        new_s = old_s - (old_s * self.kjs)
        if new_s < 0:
            new_s = 0

        new_j = old_j + (old_s * self.kjs) - (old_j * self.kgj) - (old_j * self.kjl)
        if new_j < 0:
            new_j = 0

        phi = 0 if t < self.tau else self.history[t - self.tau][j_label]
        new_l = old_l + (phi * self.kjl) - (old_l * self.kgl)
        if new_l < 0:
            new_l = 0

        g_prod = self.klambda / ((self.klambda / self.gprod0) + (old_g - self.gb))
        new_g = old_g - (self.kxg + self.kxgi * old_i) * old_g + g_prod + self.mu * (self.kgj * old_j + self.kgl * old_l)
        if new_g < 0:
            new_g = 0
        
        # g_tilde = old_g + self.fgj * (self.kgj * old_j + self.kgl * old_l)
        # new_i = old_i + self.kxi * self.ib * ((self.beta ** self.gamma + 1) / (self.beta ** self.gamma * (self.gb / g_tilde) ** self.gamma + 1) - old_i / self.ib)
        new_i = old_i + self.kxi * self.ib * ( - old_i / self.ib)

        if t in self.interventions:
            for intervention in self.interventions[t]:
                if intervention[0] == s_label:
                    new_s += intervention[1]
                elif intervention[0] == i_label:
                    new_i += intervention[1]

        timestep = {'step': t, s_label: new_s, j_label: new_j, l_label: new_l, g_label: new_g, i_label: new_i}
        self.history.append(timestep)

    def add_intervention(self, timestep: int, variable: str, intervention: float):
        if timestep not in self.interventions:
            self.interventions[timestep] = list()

        self.interventions[timestep].append((variable, intervention))

    def plot(self, timesteps = -1):
        if timesteps == -1:
            df = pd.DataFrame(self.history)
            df.plot('step', [s_label, j_label, l_label, g_label, i_label])
            plt.show()
        else:
            df = pd.DataFrame(self.history[:timesteps])
            df.plot('step', [s_label, j_label, l_label, g_label, i_label])
            plt.show()
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label
import pandas as pd
import matplotlib.pyplot as plt


class Model:
    def __init__(self, starting_vals, constants, interventions: list = None):
        self.interventions = dict()

        if interventions is not None:
            for intervention in interventions:
                self.add_intervention(*intervention)

        self.history = []
        self.history.append(
            {
                "step": 0,
                "eat": 0,
                "snack": 0,
                "light_meal": 0,
                "heavy_meal": 0,
                "bolus": 0,
                "basal": 0,
                s_label: starting_vals[0],
                j_label: starting_vals[1],
                l_label: starting_vals[2],
                g_label: starting_vals[3],
                i_label: starting_vals[4],
            }
        )

        self.kjs = constants[0]
        self.kgj = constants[1]
        self.kjl = constants[2]
        self.kgl = constants[3]
        self.kxg = constants[4]
        self.kxgi = constants[5]
        self.kxi = constants[6]

        self.tau = constants[7]
        self.klambda = constants[8]
        self.eta = constants[9]

        self.gprod0 = constants[10]
        self.kmu = constants[11]
        self.gb = starting_vals[3]

        self.gprod_limit = (self.klambda * self.gb + self.gprod0 * (self.kmu + self.gb)) / (self.klambda + self.gprod0)

    def update(self, t):
        old_s = self.history[t - 1][s_label]
        old_j = self.history[t - 1][j_label]
        old_l = self.history[t - 1][l_label]
        old_g = self.history[t - 1][g_label]
        old_i = self.history[t - 1][i_label]

        new_s = old_s - (old_s * self.kjs)

        new_j = old_j + (old_s * self.kjs) - (old_j * self.kgj) - (old_j * self.kjl)

        phi = 0 if t < self.tau else self.history[t - self.tau][j_label]
        new_l = old_l + (phi * self.kjl) - (old_l * self.kgl)

        g_prod = (
            (self.klambda * (self.gb - old_g)) / (self.kmu + (self.gb - old_g)) + self.gprod0
            if old_g <= self.gprod_limit
            else 0
        )
        new_g = (
            old_g - (self.kxg + self.kxgi * old_i) * old_g + g_prod + self.eta * (self.kgj * old_j + self.kgl * old_l)
        )

        new_i = old_i - (old_i * self.kxi)
        eat = 0
        bolus = 0
        basal = 0
        snack = 0
        light_meal = 0
        heavy_meal = 0

        if t in self.interventions:
            for intervention in self.interventions[t]:
                if intervention[0] == s_label:
                    new_s += intervention[1]
                    eat = 1
                elif intervention[0] == i_label:
                    new_i += intervention[1]
                elif intervention[0] == "bolus":
                    new_i += intervention[1]
                    bolus = 1
                elif intervention[0] == "basal":
                    new_i += intervention[1]
                    basal = 1
                elif intervention[0] == "snack":
                    new_s += intervention[1]
                    snack = 1
                elif intervention[0] == "light_meal":
                    new_s += intervention[1]
                    light_meal = 1
                elif intervention[0] == "heavy_meal":
                    new_s += intervention[1]
                    heavy_meal = 1

        timestep = {
            "step": t,
            "eat": eat,
            "snack": snack,
            "light_meal": light_meal,
            "heavy_meal": heavy_meal,
            "bolus": bolus,
            "basal": basal,
            s_label: new_s,
            j_label: new_j,
            l_label: new_l,
            g_label: new_g,
            i_label: new_i,
        }
        self.history.append(timestep)
        return timestep

    def add_intervention(self, timestep: int, variable: str, intervention: float):
        assert intervention is not None, f"{variable} at timestep {timestep} should not be none"
        if timestep not in self.interventions:
            self.interventions[timestep] = list()

        self.interventions[timestep].append((variable, intervention))

    def plot(self, timesteps=-1):
        if timesteps == -1:
            df = pd.DataFrame(self.history)
            df.plot("step", [s_label, j_label, l_label, g_label, i_label])
            plt.show()
        else:
            df = pd.DataFrame(self.history[:timesteps])
            df.plot("step", [s_label, j_label, l_label, g_label, i_label])
            plt.show()

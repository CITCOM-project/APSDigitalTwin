import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from aps_digitaltwin.model import Model
from aps_digitaltwin.openaps import OpenAPS
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label

@dataclass
class Scenario:
    initial_carbs: float
    initial_bg: float
    initial_iob: float
    timesteps: float
    level_high: float
    level_low: float
    interventions: list

    def initial_values(self):
        return [self.initial_carbs, 0, 0, self.initial_bg, self.initial_iob]
    
    def run(self, constants: list):
        model_control = Model(self.initial_values(), constants)

        for intervention in self.interventions:
            model_control.add_intervention(intervention[0], intervention[1], intervention[2])

        for t in range(1, self.timesteps + 1):
            model_control.update(t)

        control_violations = []
        for timestep in model_control.history:
            if timestep[g_label] > self.level_high:
                control_violations.append(timestep["step"])

            if timestep[g_label] < self.level_low:
                control_violations.append(timestep["step"])

        open_aps = OpenAPS()
        model_openaps = Model(self.initial_values(), constants)

        for intervention in self.interventions:
            model_openaps.add_intervention(intervention[0], intervention[1], intervention[2])

        for t in range(1, self.timesteps + 1):
            if t % 5 == 1:
                rate = open_aps.run(model_openaps.history)
                model_openaps.add_intervention(t, i_label, rate)
            model_openaps.update(t)

        openaps_violations = []
        for timestep in model_openaps.history:
            if timestep[g_label] > self.level_high:
                openaps_violations.append(timestep["step"])

            if timestep[g_label] < self.level_low:
                openaps_violations.append(timestep["step"])

        fig, (ax1, ax2) = plt.subplots(1,2)
        control_df = pd.DataFrame(model_control.history)
        control_df.plot('step', [s_label, j_label, l_label, g_label, i_label], ax=ax1)

        openaps_df = pd.DataFrame(model_openaps.history)
        openaps_df.plot('step', [s_label, j_label, l_label, g_label, i_label], ax=ax2)

        plt.show()

        return len(control_violations) <= len(openaps_violations)
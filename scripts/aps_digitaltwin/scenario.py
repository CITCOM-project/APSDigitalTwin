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
    
    def run(self, constants: list, recorded_carbs = None, output_file = None):
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

        open_aps = OpenAPS(recorded_carbs)
        model_openaps = Model(self.initial_values(), constants)
        pump_output = 0

        for intervention in self.interventions:
            model_openaps.add_intervention(intervention[0], intervention[1], intervention[2])

        for t in range(1, self.timesteps + 1):
            if t % 5 == 1:
                rate = open_aps.run(model_openaps.history)
                pump_output += rate
                for j in range(5):
                    model_openaps.add_intervention(t + j, i_label, rate / 5.0)
            model_openaps.update(t)

        hyper_violations = []
        hypo_violations = []
        bg_change = 0
        for idx, timestep in enumerate(model_openaps.history):
            if not idx == 0:
                bg_change += abs(timestep[g_label] - model_openaps.history[idx - 1][g_label])
            if timestep[g_label] > self.level_high:
                hyper_violations.append(timestep["step"])

            if timestep[g_label] < self.level_low:
                hypo_violations.append(timestep["step"])        

        if output_file == None or recorded_carbs == None:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.set_title("No Intervention")
            ax1.hlines(y=[self.level_high, self.level_low], xmin=0, xmax=self.timesteps, colors='r', linestyles='--', lw=1)
            ax2.set_title("OpenAPS Intervention")
            ax2.hlines(y=[self.level_high, self.level_low], xmin=0, xmax=self.timesteps, colors='r', linestyles='--', lw=1)

            control_df = pd.DataFrame(model_control.history)
            control_df.plot('step', [s_label, j_label, l_label, g_label, i_label], ax=ax1)

            openaps_df = pd.DataFrame(model_openaps.history)
            openaps_df.plot('step', [s_label, j_label, l_label, g_label, i_label], ax=ax2)
            
            plt.show()
        else:
            output = open(output_file, "a")
            line = [self.initial_bg, self.initial_carbs, recorded_carbs, self.initial_iob, pump_output, bg_change, len(hypo_violations), len(hyper_violations)]

            output.write(','.join(map(str,line)) + "\n")

        return len(control_violations) >= len(hyper_violations + hypo_violations)
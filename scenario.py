from dataclasses import dataclass
from model import Model
from openaps import OpenAPS
from setup import g_label
import os

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

        # model_control.plot()

        open_aps = OpenAPS()
        model_openaps = Model(self.initial_values(), constants)

        for intervention in self.interventions:
            model_openaps.add_intervention(intervention[0], intervention[1], intervention[2])

        for t in range(1, self.timesteps + 1):
            os.system("oref0-calculate-iob pumphistory.json profile.json clock.json autosens.json > iob.json")
            model_openaps.update(t)

        openaps_violations = []
        for timestep in model_openaps.history:
            if timestep[g_label] > self.level_high:
                openaps_violations.append(timestep["step"])

            if timestep[g_label] < self.level_low:
                openaps_violations.append(timestep["step"])

        # model_openaps.plot()

        return len(control_violations) <= len(openaps_violations)
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from aps_digitaltwin.model import Model
from aps_digitaltwin.openaps import OpenAPS
from aps_digitaltwin.util import s_label, j_label, l_label, g_label, i_label, BOLUS, BASAL_BOLUS_THRESHOLD


@dataclass
class Scenario:
    initial_carbs: float
    initial_bg: float
    initial_iob: float
    timesteps: float
    level_high: float
    level_low: float
    interventions: list
    initial_il: float = 0
    initial_jej: float = 0

    def initial_values(self):
        return [self.initial_carbs, self.initial_jej, self.initial_il, self.initial_bg, self.initial_iob]

    def _plot(self, df):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("No Intervention")
        ax1.hlines(y=[self.level_high, self.level_low], xmin=0, xmax=self.timesteps, colors="r", linestyles="--", lw=1)
        ax2.set_title("OpenAPS Intervention")
        ax2.hlines(y=[self.level_high, self.level_low], xmin=0, xmax=self.timesteps, colors="r", linestyles="--", lw=1)

        df.plot("step", [s_label, j_label, l_label, g_label, i_label], ax=ax2)
        plt.show()

    def run_control(
        self,
        constants: list,
        plot=False,
        kill_at_fault=False,
    ):
        model_control = Model(self.initial_values(), constants)

        for intervention in self.interventions:
            model_control.add_intervention(intervention[0], intervention[1], intervention[2])

        control_violations = []
        fault = False
        for t in range(1, self.timesteps + 1):
            timestep = model_control.update(t)
            if not (self.level_low < timestep[g_label] < self.level_high):
                fault = True
                control_violations.append(timestep["step"])
                if kill_at_fault and fault and t % timesteps_per_intervention == 0:
                    break

        control_df = pd.DataFrame(model_control.history)

        if plot:
            self._plot(control_df)
        return control_df

    def run(
        self,
        constants: list,
        recorded_carbs=None,
        plot=False,
        model_control=False,
        tempdir="openaps_temp",
        profile_path=None,
        basal_profile_path=None,
        kill_at_fault=False,
        timesteps_per_intervention=5,
    ):
        if model_control:
            control_df = run_control(constants, plot, kill_at_fault)

        open_aps = OpenAPS(recorded_carbs, profile_path=profile_path, basal_profile_path=basal_profile_path)
        model_openaps = Model(self.initial_values(), constants, interventions=self.interventions)

        fault = False
        for t in range(1, self.timesteps + 1):
            if t % timesteps_per_intervention == 0:
                rate = open_aps.run(model_openaps.history, tempdir)
                if rate > BASAL_BOLUS_THRESHOLD:
                    model_openaps.add_intervention(t, "bolus", BOLUS)
                else:
                    for j in range(5):
                        model_openaps.add_intervention(t + j, i_label, rate / 5.0)
            timestep = model_openaps.update(t)
            fault = fault or (not (self.level_low < timestep[g_label] < self.level_high))
            if kill_at_fault and fault and t % timesteps_per_intervention == 0:
                break
        openaps_df = pd.DataFrame(model_openaps.history)
        openaps_df["Safe"] = openaps_df["Blood Glucose"].between(self.level_low, self.level_high)

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

        if plot:
            self._plot(openaps_df)
        return openaps_df

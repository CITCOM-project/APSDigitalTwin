"""
Constants and training data processing.
"""

import pandas as pd
from scipy.stats import beta


s_label = "Stomach"
j_label = "Jejunum"
l_label = "Ileum"
g_label = "Blood Glucose"
i_label = "Blood Insulin"

BASAL_MIN = 5
BASAL_BOLUS_THRESHOLD = 35
BASAL = 20
BOLUS = 45
STOMACH = 100

SNACK = 80
LIGHT_MEAL = 171
HEAVY_MEAL = 243


LOW = 50
HIGH = 180

constant_names = ["kjs", "kgj", "kjl", "kgl", "kxg", "kxgi", "kxi", "τ", "η", "kλ", "kμ", "Gprod0"]
intervention_values = {"bolus": BOLUS, "snack": SNACK, "light_meal": LIGHT_MEAL, "heavy_meal": HEAVY_MEAL}

beta_bg = beta(4.41173978523873, 93778.91739596258, 33.94794766515366, 2070111.880383831)
beta_iob = beta(1.641356889981472, 5.509053842226525, -1.0868011668755448, 22.072273745357563)
beta_cob = beta(2.2471512058441854, 6.84827995821931, -0.9388421899263029, 145.35307806549673)


class TrainingData:
    def __init__(self, data_path=None, timesteps=24, data_frame=None) -> None:
        self.timesteps = timesteps
        self.counter = 0

        if data_frame is None:
            data_frame = pd.read_csv(data_path)
        data_frame = data_frame[: self.timesteps]

        assert (
            len(data_frame) == timesteps
        ), f"Dataframe with {len(data_frame)} elements did not match {timesteps} timesteps"
        self.bg_data_frame = self.interpolate_dataset(data_frame["bg"])
        self.iob_data_frame = self.interpolate_dataset(
            1000 * data_frame["iob"] / 60.0
        )  # units / hour to units / minute
        self.cob_data_frame = self.interpolate_dataset(1000 * data_frame["cob"] / 180.156)  # Correct units

        self.interventions = self._find_interventions(data_frame)

    def find_initial_values(self):
        inital_bg = self.bg_data_frame[0]
        inital_insulin = self.iob_data_frame[0]
        initial_stomach = self.cob_data_frame[0]

        return [initial_stomach, 0, 0, inital_bg, inital_insulin]

    def _find_interventions(self, data_frame):
        interventions = []

        rates_dataframe = data_frame["rate"]
        for i, rate in rates_dataframe.items():
            for j in range(5):
                interventions.append((i * 5 + j, i_label, (1000 * rate / 60.0) / 5.0))

        for i in range(len(self.cob_data_frame) - 1):
            diff = self.cob_data_frame[i + 1] - self.cob_data_frame[i]
            if diff > 0:
                interventions.append((i + 1, s_label, diff))

        return interventions

    def interpolate_dataset(self, data_series: pd.Series):
        temp_list = list(data_series[:1])

        for i in range(len(data_series) - 1):
            diff = (data_series[i + 1] - data_series[i]) / 5.0
            for j in range(5):
                temp_list.append(data_series[i] + j * diff)

        new_series = pd.Series(temp_list)

        return new_series

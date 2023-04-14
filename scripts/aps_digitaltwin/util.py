import pandas as pd

s_label = 'Stomach'
j_label = 'Jejunum'
l_label = 'Ileum'
g_label = 'Blood Glucose'
i_label = 'Blood Insulin'

class TrainingData:

    def __init__(self, data_path, timesteps = 24) -> None:
        self.timesteps = timesteps
        self.counter = 0
        
        data_frame = pd.read_csv(data_path)[:self.timesteps]
        self.bg_data_frame = self.interpolate_dataset(data_frame["bg"])
        self.iob_data_frame = self.interpolate_dataset(1000 * data_frame["iob"] / 60.0) # units / hour to units / minute
        self.cob_data_frame = self.interpolate_dataset(1000 * data_frame["cob"] / 180.156) # Correct units

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
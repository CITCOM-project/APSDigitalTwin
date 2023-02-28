import os
import subprocess
import shutil
import json
from datetime import datetime
from aps_digitaltwin.util import g_label, s_label, i_label

class OpenAPS:

    def __init__(self, profile_path, basal_profile_path, autosense_ratio = 1.0, test_timestamp = "2023-01-01T18:00:00-00:00") -> None:
        oref_help = subprocess.check_output(["oref0","--help"])

        if "oref0 help - this message" not in str(oref_help):
            print("ERROR - oref0 not installed")
            exit(1)

        self.profile_path = profile_path
        self.basal_profile_path = basal_profile_path
        self.autosense_ratio = autosense_ratio
        self.test_timestamp = test_timestamp
        self.epoch_time = int(datetime.strptime(test_timestamp, "%Y-%m-%dT%H:%M:%S%z").timestamp() * 1000)
        self.pump_history = []

    def run(self, model_history):
        if not os.path.exists('./openaps_temp'):
            os.mkdir("./openaps_temp")

        time_since_start = len(model_history) - 1
        current_epoch = self.epoch_time + 60000 * time_since_start
        current_timestamp = datetime.fromtimestamp(current_epoch / 1000).strftime("%Y-%m-%dT%H:%M:%S%z")

        basal_history = []
        temp_basal = '{}'
        if model_history[0][i_label] > 0:
            basal_history.append(f'{{"timestamp":"{datetime.fromtimestamp(self.epoch_time/1000).strftime("%Y-%m-%dT%H:%M:%S%z")}"' +
                                 f',"_type":"Bolus","amount":{model_history[0][i_label] / 1000},"duration":0}}')

        for idx, (rate, duration, timestamp) in enumerate(self.pump_history):
            basal_history.append(f'{{"timestamp":"{timestamp}","_type":"TempBasal","temp":"absolute","rate":{str(rate)}}}')
            basal_history.append(f'{{"timestamp":"{timestamp}","_type":"TempBasalDuration","duration (min)":{str(duration)}}}')
            if idx == len(self.pump_history) - 1:
                temp_basal = f'{{"duration": {duration}, "temp": "absolute", "rate": {str(rate)}}}'
        basal_history.reverse()

        glucose_history = []
        carb_history = []
        for idx, time_step in enumerate(model_history):
            if idx % 5 == 0:
                bg_level = int(time_step[g_label])
                new_time_epoch = self.epoch_time + idx * 60000
                new_time_stamp = datetime.fromtimestamp(new_time_epoch/1000).strftime("%Y-%m-%dT%H:%M:%S%z")
                glucose_history.append(f'{{"date":{new_time_epoch},"dateString":"{new_time_stamp}","sgv":{bg_level},' +
                                       f'"device":"fakecgm","type":"sgv","glucose":{bg_level}}}')
                
            if idx == 0:
                if time_step[s_label] > 0:
                    carb_history.append(f'{{"enteredBy":"fakecarbs","carbs":{time_step[s_label]},"created_at":"{self.test_timestamp}","insulin": null}}')

            else:
                carb_diff = time_step[s_label] - model_history[idx - 1][s_label]
                if carb_diff > 0:
                    new_time_epoch = self.epoch_time + idx * 60000
                    new_time_stamp = datetime.fromtimestamp(new_time_epoch/1000).strftime("%Y-%m-%dT%H:%M:%S%z")
                    carb_history.append(f'{{"enteredBy":"fakecarbs","carbs":{time_step[s_label]},"created_at":"{new_time_stamp}","insulin":null}}')
        glucose_history.reverse()
        carb_history.reverse()

        self.__make_file_and_write_to("./openaps_temp/clock.json", f'"{current_timestamp}-00:00"')
        self.__make_file_and_write_to("./openaps_temp/autosens.json", '{"ratio":' + str(self.autosense_ratio) + '}')
        self.__make_file_and_write_to("./openaps_temp/pumphistory.json", "[" + ','.join(basal_history) + "]")
        self.__make_file_and_write_to("./openaps_temp/glucose.json", "[" + ','.join(glucose_history) + "]")
        self.__make_file_and_write_to("./openaps_temp/carbhistory.json", "[" + ','.join(carb_history) + "]")
        self.__make_file_and_write_to("./openaps_temp/temp_basal.json", temp_basal)

        iob_output = subprocess.check_output([
            "oref0-calculate-iob",
            "./openaps_temp/pumphistory.json",
            self.profile_path,
            "./openaps_temp/clock.json",
            "./openaps_temp/autosens.json"
        ]).decode("utf-8")
        self.__make_file_and_write_to("./openaps_temp/iob.json", iob_output)

        meal_output = subprocess.check_output([
            "oref0-meal",
            "./openaps_temp/pumphistory.json",
            self.profile_path,
            "./openaps_temp/clock.json",
            "./openaps_temp/glucose.json",
            self.basal_profile_path,
            "./openaps_temp/carbhistory.json"
        ]).decode("utf-8")
        self.__make_file_and_write_to("./openaps_temp/meal.json", meal_output)

        suggested_output = subprocess.check_output([
            "oref0-determine-basal",
            "./openaps_temp/iob.json",
            "./openaps_temp/temp_basal.json",
            "./openaps_temp/glucose.json",
            self.profile_path,
            "--auto-sens",
            "./openaps_temp/autosens.json",
            "--meal",
            "./openaps_temp/meal.json",
            "--microbolus",
            "--currentTime",
            str(current_epoch)
        ]).decode("utf-8")
        self.__make_file_and_write_to("./openaps_temp/suggested.json", suggested_output)

        json_output = open("./openaps_temp/suggested.json")
        data = json.load(json_output)

        rate = data["rate"] if "rate" in data else 0
        if rate != 0:
            duration = data["duration"]
            timestamp = data["deliverAt"]
            self.pump_history.append((rate, duration, timestamp))

        shutil.rmtree("./openaps_temp")

        return 1000 * rate / 60.0

    def __make_file_and_write_to(self, file_path, contents):
        file = open(file_path, "w")
        file.write(contents)

if __name__ == "__main__":
    x = OpenAPS("./oref0_data/profile.json")
    x.run()
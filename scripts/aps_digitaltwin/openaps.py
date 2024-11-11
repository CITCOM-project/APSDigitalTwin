import os
import platform
import subprocess
import shutil
import json
from datetime import datetime
from aps_digitaltwin.util import g_label, s_label, i_label


class OpenAPS:
    def __init__(
        self,
        recorded_carbs=None,
        autosense_ratio=1.0,
        test_timestamp="2023-01-01T18:00:00-00:00",
        profile_path=None,
        basal_profile_path=None,
    ) -> None:
        self.shell = "Windows" in platform.system()
        oref_help = subprocess.check_output(["oref0", "--help"], shell=self.shell)

        if "oref0 help - this message" not in str(oref_help):
            print("ERROR - oref0 not installed")
            exit(1)

        if profile_path is None:
            self.profile_path = os.environ["profile_path"]
        else:
            self.profile_path = profile_path

        if basal_profile_path is None:
            self.basal_profile_path = os.environ["basal_profile_path"]
        else:
            self.basal_profile_path = basal_profile_path
        self.autosense_ratio = autosense_ratio
        self.test_timestamp = test_timestamp
        self.epoch_time = int(datetime.strptime(test_timestamp, "%Y-%m-%dT%H:%M:%S%z").timestamp() * 1000)
        self.pump_history = []
        self.recorded_carbs = recorded_carbs

    def run(self, model_history, output_file=None):
        if output_file == None:
            output_file = "./openaps_temp"

        if not os.path.exists(output_file):
            os.makedirs(output_file)

        time_since_start = len(model_history) - 1
        current_epoch = self.epoch_time + 60000 * time_since_start
        current_timestamp = datetime.fromtimestamp(current_epoch / 1000).strftime("%Y-%m-%dT%H:%M:%S%z")

        basal_history = []
        temp_basal = "{}"
        if model_history[0][i_label] > 0:
            basal_history.append(
                f'{{"timestamp":"{datetime.fromtimestamp(self.epoch_time/1000).strftime("%Y-%m-%dT%H:%M:%S%z")}"'
                + f',"_type":"Bolus","amount":{model_history[0][i_label] / 1000},"duration":0}}'
            )

        for idx, (rate, duration, timestamp) in enumerate(self.pump_history):
            basal_history.append(
                f'{{"timestamp":"{timestamp}","_type":"TempBasal","temp":"absolute","rate":{str(rate)}}}'
            )
            basal_history.append(
                f'{{"timestamp":"{timestamp}","_type":"TempBasalDuration","duration (min)":{str(duration)}}}'
            )
            if idx == len(self.pump_history) - 1:
                temp_basal_epoch = int(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").timestamp() * 1000)
                if (current_epoch - temp_basal_epoch) / 60 <= duration:
                    temp_basal = f'{{"duration": {duration}, "temp": "absolute", "rate": {str(rate)}}}'
        basal_history.reverse()

        glucose_history = []
        carb_history = []
        for idx, time_step in enumerate(model_history):
            if idx % 5 == 0:
                bg_level = int(time_step[g_label])
                new_time_epoch = self.epoch_time + idx * 60000
                new_time_stamp = datetime.fromtimestamp(new_time_epoch / 1000).strftime("%Y-%m-%dT%H:%M:%S%z")
                glucose_history.append(
                    f'{{"date":{new_time_epoch},"dateString":"{new_time_stamp}","sgv":{bg_level},'
                    + f'"device":"fakecgm","type":"sgv","glucose":{bg_level}}}'
                )

            if idx == 0:
                if time_step[s_label] > 0:
                    if self.recorded_carbs == None:
                        carb_history.append(
                            f'{{"enteredBy":"fakecarbs","carbs":{time_step[s_label]},"created_at":"{self.test_timestamp}","insulin": null}}'
                        )
                    else:
                        carb_history.append(
                            f'{{"enteredBy":"fakecarbs","carbs":{self.recorded_carbs},"created_at":"{self.test_timestamp}","insulin": null}}'
                        )

            else:
                carb_diff = time_step[s_label] - model_history[idx - 1][s_label]
                if carb_diff > 0:
                    new_time_epoch = self.epoch_time + idx * 60000
                    new_time_stamp = datetime.fromtimestamp(new_time_epoch / 1000).strftime("%Y-%m-%dT%H:%M:%S%z")
                    carb_history.append(
                        f'{{"enteredBy":"fakecarbs","carbs":{time_step[s_label]},"created_at":"{new_time_stamp}","insulin":null}}'
                    )
        glucose_history.reverse()
        carb_history.reverse()

        self.__make_file_and_write_to(f"{output_file}/clock.json", f'"{current_timestamp}-00:00"')
        self.__make_file_and_write_to(f"{output_file}/autosens.json", '{"ratio":' + str(self.autosense_ratio) + "}")
        self.__make_file_and_write_to(f"{output_file}/pumphistory.json", "[" + ",".join(basal_history) + "]")
        self.__make_file_and_write_to(f"{output_file}/glucose.json", "[" + ",".join(glucose_history) + "]")
        self.__make_file_and_write_to(f"{output_file}/carbhistory.json", "[" + ",".join(carb_history) + "]")
        self.__make_file_and_write_to(f"{output_file}/temp_basal.json", temp_basal)

        iob_output = subprocess.check_output(
            [
                "oref0-calculate-iob",
                f"{output_file}/pumphistory.json",
                self.profile_path,
                f"{output_file}/clock.json",
                f"{output_file}/autosens.json",
            ],
            shell=self.shell,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        self.__make_file_and_write_to(f"{output_file}/iob.json", iob_output)

        meal_output = subprocess.check_output(
            [
                "oref0-meal",
                f"{output_file}/pumphistory.json",
                self.profile_path,
                f"{output_file}/clock.json",
                f"{output_file}/glucose.json",
                self.basal_profile_path,
                f"{output_file}/carbhistory.json",
            ],
            shell=self.shell,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        self.__make_file_and_write_to(f"{output_file}/meal.json", meal_output)

        suggested_output = subprocess.check_output(
            [
                "oref0-determine-basal",
                f"{output_file}/iob.json",
                f"{output_file}/temp_basal.json",
                f"{output_file}/glucose.json",
                self.profile_path,
                "--auto-sens",
                f"{output_file}/autosens.json",
                "--meal",
                f"{output_file}/meal.json",
                "--microbolus",
                "--currentTime",
                str(current_epoch),
            ],
            shell=self.shell,
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
        self.__make_file_and_write_to(f"{output_file}/suggested.json", suggested_output)

        # self.__update_suggested(suggested_output, "./suggested_list.json")

        json_output = open(f"{output_file}/suggested.json")
        data = json.load(json_output)

        rate = data["rate"] if "rate" in data else 0
        if rate != 0:
            duration = data["duration"]
            timestamp = data["deliverAt"]
            self.pump_history.append((rate, duration, timestamp))

        shutil.rmtree(output_file, ignore_errors=True)

        return 1000 * rate / 60.0

    def __make_file_and_write_to(self, file_path, contents):
        file = open(file_path, "w")
        file.write(contents)

    def __update_suggested(self, suggested_json, file_path):
        if not os.path.isfile(file_path):
            new_file = open(file_path, "w")
            new_file.write("[]")
            new_file.close()

        list_json = None
        with open(file_path) as output_file:
            list_json = json.load(output_file)
            list_json.append(json.loads(suggested_json))

        with open(file_path, "w") as writing_json:
            json.dump(list_json, writing_json, indent=4, separators=(",", ": "))

import os
import subprocess
import shutil

class OpenAPS:

    def __init__(self, profile_path, autosense_ratio = 1.0, test_timestamp = "2023-01-01T18:00:00-0000") -> None:
        oref_help = subprocess.check_output(["oref0","--help"])

        if "oref0 help - this message" not in str(oref_help):
            print("ERROR - oref0 not installed")
            exit(1)

        self.profile_path = profile_path
        self.autosense_ratio = autosense_ratio
        self.test_timestamp = test_timestamp

        self.pump_history = []

    def run(self):
        if not os.path.exists('./openaps_temp'):
            os.mkdir("./openaps_temp")

        self.__make_file_and_write_to("./openaps_temp/clock.json", '"' + self.test_timestamp + '"')
        self.__make_file_and_write_to("./openaps_temp/autosens.json", '{"ratio":' + str(self.autosense_ratio) + '}')
        self.__make_file_and_write_to("./openaps_temp/pumphistory.json", "[]")

        iob_output = subprocess.check_output([
            "oref0-calculate-iob",
            "./openaps_temp/pumphistory.json",
            self.profile_path,
            "./openaps_temp/clock.json",
            "./openaps_temp/autosens.json"])
        
        self.__make_file_and_write_to("./openaps_temp/iob.json", iob_output)

        shutil.rmtree("./openaps_temp")

    def __make_file_and_write_to(self, file_path, contents):
        file = open(file_path, "w")
        file.write(contents)

if __name__ == "__main__":
    x = OpenAPS("./oref0_data/profile.json")
    x.run()
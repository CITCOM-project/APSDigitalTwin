import sys
import os
import pandas as pd
from datetime import datetime
import time
import math

def process_devstat_contents(file_path):
    df = pd.read_csv(file_path)
    intervention_dict = dict()

    if "openaps/iob/iob" in df and ("openaps/suggested/COB" in df or "openaps/enacted/COB" in df) and ("openaps/suggested/rate" in df or "openaps/enacted/rate" in df):
        for idx, row in df.iterrows():
            if row["openaps/iob/iob"] != "":
                iob = row["openaps/iob/iob"]
                cob = None
                if "openaps/suggested/COB" in row and not math.isnan(row["openaps/suggested/COB"]):
                    cob = row["openaps/suggested/COB"]
                elif "openaps/enacted/COB" in row and not math.isnan(row["openaps/enacted/COB"]):
                    cob = row["openaps/enacted/COB"]
                else:
                    continue

                rate = 0
                if "openaps/suggested/rate" in row and not math.isnan(row["openaps/suggested/rate"]):
                    rate = row["openaps/suggested/rate"]
                elif "openaps/enacted/rate" in row and not math.isnan(row["openaps/enacted/rate"]):
                    rate = row["openaps/enacted/rate"]

                timestamp = None
                try:
                    timestamp = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                except:
                    try:
                        timestamp = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%S%Z")
                    except:
                        try:
                            timestamp = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ")
                        except:
                            try:
                                timestamp = datetime.strptime(row["created_at"], "%Y-%m-%dT%H:%M:%S.%f%Z")
                            except:
                                continue
                epoch = time.mktime(timestamp.timetuple())

                intervention_dict[epoch] = {"iob": iob, "cob": cob, "rate": rate}

    return intervention_dict


def process_entries_contents(file_path, interventions, person_output):
    df = pd.read_csv(file_path)

    output = None
    if not os.path.exists(person_output):
        output = open(person_output, "w")
        output.write("timestamp,bg,iob,cob,rate\n")
    else:
        output = open(person_output, "a")

    outputs = []

    for idx, row in df.iterrows():
        timestamp = None
        try:
            timestamp = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%SZ")
        except:
            try:
                timestamp = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S%Z")
            except:
                try:
                    timestamp = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%fZ")
                except:
                    try:
                        timestamp = datetime.strptime(row[0], "%Y-%m-%dT%H:%M:%S.%f%Z")
                    except:
                        continue

        epoch = time.mktime(timestamp.timetuple())
        bg = row[1]

        index = min(interventions.keys(), key=lambda x:abs(x-epoch))
        intervention = interventions[index]
        iob = intervention["iob"]
        cob = intervention["cob"]
        rate = intervention["rate"]

        outputs.append(f"{epoch},{bg},{iob},{cob},{rate}\n")

    outputs.reverse()
    output.writelines(outputs)
    output.close()


def split_output(output_dir, output_file):
    print(f"Splitting {output_file[:-4]}")
    file = os.path.join(output_dir, output_file)
    df = pd.read_csv(file)

    split_dir = os.path.join(output_dir, "split")

    num_files = 1

    zeros = 0
    last_cob = 0
    current = []
    for idx, row in df.iterrows():
        eaten = row['cob'] > last_cob
        last_cob = row['cob']
        if zeros >= 24 and (row['cob'] != 0.0 or current != []):
            if eaten and current != []:
                zeros = 0
                current = []
                continue
            current.append(f"{row['timestamp']},{row['bg']},{row['iob']},{row['cob']},{row['rate']}\n")
            if len(current) == 48:
                if not os.path.exists(split_dir):
                    os.mkdir(split_dir)
                output_file_name = os.path.join(split_dir, f"{output_file[:-4]}_{num_files}.csv")
                output = open(output_file_name, "w")
                output.write("timestamp,bg,iob,cob,rate\n")
                output.writelines(current)
                output.close()

                zeros = 0
                current = []
                num_files += 1
        elif row['cob'] == 0.0:
            zeros += 1
        

dir_path = sys.argv[1]
output_dir = sys.argv[2]
print(f"Scanning dir {dir_path}...")

for filename in os.listdir(dir_path):
    print(f"Processing: {filename}")
    person_file = os.path.join(dir_path, filename, "direct-sharing-31")

    interventions = dict()
    person_output = os.path.join(output_dir, filename + ".csv")

    for internal in os.listdir(person_file):
        intern = os.path.join(person_file, internal)

        if os.path.isdir(intern) and "devicestatus" in internal and "parts" not in internal:
            for devstat in os.listdir(intern):
                if devstat.endswith(".csv"):
                    intervention_dict = process_devstat_contents(os.path.join(intern, devstat))
                    interventions.update(intervention_dict)

    if len(interventions) != 0:
        for internal in os.listdir(person_file):
            intern = os.path.join(person_file, internal)

            if os.path.isdir(intern) and "entries" in internal:
                for entries in os.listdir(intern):
                    if entries.endswith(".csv"):
                        process_entries_contents(os.path.join(intern, entries), interventions, person_output)
    else:
        print(f"Rejecting {filename} - IC2")

for output_file in os.listdir(output_dir):
    if output_file.endswith(".csv"):
        file = open(os.path.join(output_dir, output_file))
        content = file.read()
        file.close()
        if "nan" in content or "null" in content:
            print(f"Removing {output_file[:-4]} - IC2")
            os.remove(os.path.join(output_dir, output_file))

for output_file in os.listdir(output_dir):
    if output_file.endswith(".csv"):
        split_output(output_dir, output_file)
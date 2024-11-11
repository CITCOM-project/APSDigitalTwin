"""
This module concatenates the JSON log files from individual simulator runs into
a single JSON file which contains all of the attacks in a format which can be
interpreted for g-estimation of the causal effect of individual interventions.
"""
from glob import glob
import json
from os.path import exists
import pandas as pd
import matplotlib.pyplot as plt

attacks = []
attack_lengths = []
trace_lengths = []
full_trace_lengths = []

for fname in glob("test/*.json"):
    with open(fname) as f:
        attack = json.load(f)

    attack_trace = pd.DataFrame(attack["attack"], columns=["timestep", "variable", "value"])
    minimal = pd.DataFrame(attack["minimal"], columns=["timestep", "variable", "value"])

    attack_trace["value"] = attack_trace["value"].astype(bool).astype(int)
    minimal["value"] = minimal["value"].astype(bool).astype(int)

    if len(minimal) > 0 and exists(fname.replace(".json", "_minimal_fault.csv")):
        t_start = minimal["timestep"].min() - 5

        df = pd.read_csv(fname.replace(".json", "_minimal_fault.csv"), index_col=0)
        full_trace_lengths.append(len(df) / 5)
        df = df.loc[df["step"] >= t_start]

        attack["initial_carbs"] = df["Stomach"][t_start]
        attack["initial_jej"] = df["Jejunum"][t_start]
        attack["initial_il"] = df["Ileum"][t_start]
        attack["initial_iob"] = df["Blood Insulin"][t_start]
        attack["initial_bg"] = df["Blood Glucose"][t_start]
        attack["outcome"] = attack["outcome"].replace(" ", "_")

        attack["attack"] = attack_trace.to_records(index=False).tolist()
        attack["minimal"] = minimal.to_records(index=False).tolist()

        attacks.append(attack)

        attack_lengths.append(len(minimal))
        trace_lengths.append(len(df) / 5)

with open("successful_attacks.json", "w") as f:
    json.dump(attacks, f)

plt.hist(attack_lengths)
plt.title("Attack lengths")
plt.savefig("/tmp/attack_lengths.png")

plt.clf()
plt.hist(trace_lengths, bins=25)
plt.title("Trace lengths")
plt.savefig("/tmp/trace_lengths.png")

plt.clf()
plt.hist(full_trace_lengths, bins=25)
plt.title("Trace lengths")
plt.savefig("/tmp/full_trace_lengths.png")

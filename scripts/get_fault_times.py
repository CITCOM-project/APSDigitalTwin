import json
from uuid import uuid4
from multiprocessing import Pool
from numpy import ceil
from dotenv import load_dotenv
import pandas as pd
from glob import glob

from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario
from aps_digitaltwin.util import (
    LOW,
    HIGH,
    BASAL_BOLUS_THRESHOLD,
    BOLUS,
    SNACK,
    LIGHT_MEAL,
    HEAVY_MEAL,
)
from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm

load_dotenv()


def process_attack(inx, f):
    with open(f) as f:
        attack = json.load(f)
    attack["index"] = inx
    s1 = Scenario(
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        timesteps=5000,
        level_high=HIGH,
        level_low=LOW,
        interventions=attack["attack"],
    )
    sim_df = s1.run(attack["constants"], tempdir=f"tmp/{uuid4().hex}", kill_at_fault=True)
    sim_df.to_csv(f"successful_attacks/{inx}.csv")
    if not ~(sim_df["Safe"]).all():
        print(f"No fault for attack {inx}: {f}")
        attack["fault_time"] = None
    else:
        attack["fault_time"] = int(sim_df.loc[~sim_df["Safe"], "step"].min())
    return attack


with Pool() as pool:
    successful_attacks = pool.starmap(process_attack, enumerate(glob("test_jsons/*.json")))

with open("successful_attacks_timings.json", "w") as f:
    json.dump(sorted(successful_attacks, key=lambda x: x["index"]), f)

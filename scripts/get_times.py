from os.path import basename
import json
from uuid import uuid4
from multiprocessing import Pool
from numpy import ceil
from dotenv import load_dotenv
import pandas as pd

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


with open("successful_attacks.json") as f:
    attacks = json.load(f)


def process_attack(inx, attack):
    s1 = Scenario(
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        timesteps=2000,
        level_high=HIGH,
        level_low=LOW,
        interventions=attack["attack"],
    )
    sim_df = s1.run(attack["constants"], tempdir=f"tmp/{uuid4().hex}", kill_at_fault=True)
    sim_df.to_csv(f"successful_attacks/{inx}.csv")
    attack["fault_time"] = sim_df.loc[~sim_df["Safe"], "step"].max()
    attack["index"] = inx
    return attack


with Pool() as pool:
    attacks = pool.starmap(process_attack, enumerate(attacks))

with open("successful_attacks_timings.json", "w") as f:
    json.dump(attacks, f)

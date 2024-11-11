"""
This module finds sequences of interventions from the OpenAPS data that lead to
unsafe states. These sequences are minimised in a greedy way by iteratively
removing events from the head of the trace and rerunning the simulator to see
if the fault still occurs.
"""

from os.path import basename
import json
from uuid import uuid4
from multiprocessing import Pool
from glob import glob
from itertools import product
import sys

from numpy import ceil
from dotenv import load_dotenv
import pandas as pd

from aps_digitaltwin.util import TrainingData
from aps_digitaltwin.scenario import Scenario
from aps_digitaltwin.util import LOW, HIGH, BASAL_BOLUS_THRESHOLD, BOLUS, SNACK, LIGHT_MEAL, HEAVY_MEAL, constant_names
from aps_digitaltwin.genetic import GlucoseInsulinGeneticAlgorithm

load_dotenv()


def round_up(x: float, base: int = 5):
    """
    Round the input to the nearest multiple of `base`.

    :param x: The number to round.
    :param base: The base to round to.

    :return: The rounded value.
    """
    return base * round(x / base)


def round_to_closest(x: float, values: list) -> int:
    """
    Round the input to the nearest value in `values`.

    :param x: The number to round.
    :param values: The candidate values to round to.

    :return: The member of `values` which is closest to `x`.
    """
    return min(values, key=lambda x2: abs(x2 - x))


# pylint: disable=R0913,R0914
def reproduce_fault(
    timesteps: int,
    initial_carbs: float,
    initial_bg: float,
    initial_iob: float,
    interventions: list,
    constants: list,
    save_path: str = None,
) -> (bool, int):
    """
    Attempt to reproduce a fault from the dataset by executing the given
    interventions for the given number of timesteps.

    :param timesteps: The number of timesteps to run the simulator for.
    :param initial_carbs: The initial carbohydrates in the stomach.
    :param initial_bg: The initial blood glucose.
    :param initial_iob: The initial amount of insulin in the blood.
    :param interventions: The interventions - a list of triples of the form (time, variable, value).
    :param constants: The subject's medical constants. See documentation of `scenario.run` for further details.
    :param save_path: Filepath to save the run details as a JSON file (optional).

    :return: Whether or not a fault occurred.
    """
    s1 = Scenario(
        initial_carbs=initial_carbs,
        initial_bg=initial_bg,
        initial_iob=initial_iob,
        timesteps=timesteps,
        level_high=HIGH,
        level_low=LOW,
        interventions=interventions,
    )
    sim_df = s1.run(constants, tempdir=f"tmp/{uuid4().hex}", kill_at_fault=True)
    for k, v in zip(constant_names, constants):
        sim_df[k] = v
    if save_path is not None:
        sim_df.to_csv(save_path)

    if sim_df["Safe"].all():
        return (False, None)

    unsafe = sim_df.loc[~sim_df["Safe"], "step"]
    fault_time = int(unsafe.min())
    glucose_value = sim_df.loc[fault_time, "Blood Glucose"]
    fault = "Low" if glucose_value < LOW else "High" if glucose_value > HIGH else None
    assert fault is not None
    return fault, fault_time


def negate(timestep: int, label: str, value: int) -> tuple:
    """
    Negate a given intervention.

    :param timestep: The timestep of the intervention.
    :param label: The name of the variable being intervened on.
    :param value: The value of the variable to be negated (either 1 or 0).

    :return: A triple of the original arguments, with `value` negated.
    """
    return (timestep, label, int(not value))


# pylint: disable=R0914,R0913
def minimise_attack(group: pd.DataFrame, save_path: str, minutes=500):
    """
    Find the minimal set of interventions that still causes the system to fail
    in the same way. I.e. if the original failure was a hypo, the failure caused
    by the minimal set of interventions should still be a hypo.

    :param group: Dataframe representing the attack trace.
    :param save_path: Where to save the resulting log.
    :param minutes: How many time steps to run the simulation for (defaults to 500).
    """
    group["timestep"] = group.index * 5
    group["cob_diff"] = group["sim_cob"].diff()
    df_increases = group.loc[(group["cob_diff"] > 0) & (group["timestep"] > 0)]
    meals = df_increases[["timestep", "cob_diff"]].copy()
    meals["value"] = [round_to_closest(x, (SNACK, LIGHT_MEAL, HEAVY_MEAL)) for x in meals["cob_diff"]]
    meals["label"] = [
        "snack" if x == SNACK else "light_meal" if x == LIGHT_MEAL else "heavy_meal" for x in meals["value"]
    ]

    insulin = group.loc[(group["sim_rate"] > BASAL_BOLUS_THRESHOLD) & (group["timestep"] > 0), ["timestep", "sim_rate"]]
    insulin["value"] = BOLUS
    insulin["label"] = "bolus"

    interventions = (
        pd.concat(
            [
                meals[["timestep", "label", "value"]],
                insulin[["timestep", "label", "value"]],
            ]
        )
        .to_records(index=False)
        .tolist()
    )
    interventions.sort()

    ga = GlucoseInsulinGeneticAlgorithm()
    # Run the GA to get the best configuration. Somers et. al. ran this 10 times, but this takes a very long time and we
    # are less concerned with clinical accuracy here. Unfortunately, the GA is nondeterministic and cannot be made
    # deterministic by fixing a random seed.
    constants, _ = ga.run(TrainingData(data_frame=group, timesteps=min(24, len(group))))

    initial_carbs = group.iloc[0]["cob"]
    initial_bg = group.iloc[0]["bg"]
    initial_iob = group.iloc[0]["iob"]

    # Run the full set of interventions to see if the fault can be reproduced on the simulator
    failure, fault_time = reproduce_fault(
        timesteps=minutes - 1,
        initial_carbs=initial_carbs,
        initial_bg=initial_bg,
        initial_iob=initial_iob,
        constants=constants,
        interventions=interventions,
        save_path=save_path,
    )

    # Run without interventions to see if the run is doomed
    doomed, _ = reproduce_fault(
        timesteps=minutes - 1,
        initial_carbs=initial_carbs,
        initial_bg=initial_bg,
        initial_iob=initial_iob,
        constants=constants,
        interventions=[],
        save_path=save_path.replace(".csv", "_empty.csv"),
    )

    # Trim the interventions down to just those that happen before the (first) failure, since these can be the only
    # possible causes of the failure
    if fault_time is not None:
        interventions = [(t, var, val) for t, var, val in interventions if t < fault_time]

    if failure and not doomed and len(interventions) > 0:
        minimal = dict(enumerate(interventions))
        spurious = {}
        # Iteratively invert interventions, starting with the first one, keeping only those for which the failure does
        # not occur with the inverted value.
        for k in range(len(interventions)):
            failure_1, _ = reproduce_fault(
                timesteps=minutes - 1,
                initial_carbs=initial_carbs,
                initial_bg=initial_bg,
                initial_iob=initial_iob,
                constants=constants,
                interventions=[minimal[k2] if k2 != k else negate(*minimal[k2]) for k2 in sorted(list(minimal.keys()))],
                save_path=save_path.replace(".csv", "_minimal.csv"),
            )
            if failure_1 == failure:
                spurious[k] = minimal.pop(k)

        # Further minimise the trace by considering all combinations of the remaining interventions, starting with the
        # minimum number of interventions and gradually working back up. This step is necessary because the greedy
        # method above can sometimes mark spurious events as necessary causes, presumably due to interactions between
        # interventions. However, it is too expensive to start off simply by considering combinations.
        minimal_keys = sorted(list(minimal.keys()))
        for mask in sorted(list(product([0, 1], repeat=len(minimal))), key=sum)[1:]:
            candidate = [minimal[k] for m, k in zip(mask, minimal_keys) if m]
            failure_1, _ = reproduce_fault(
                timesteps=minutes - 1,
                initial_carbs=initial_carbs,
                initial_bg=initial_bg,
                initial_iob=initial_iob,
                constants=constants,
                interventions=candidate,
                save_path=save_path.replace(".csv", "_minimal.csv"),
            )
            if failure_1 == failure:
                minimal = candidate
                break

        with open(save_path.replace(".csv", ".json"), "w") as outfile:
            json.dump(
                {
                    "attack": interventions,
                    "timesteps": minutes - 1,
                    "fault_time": fault_time,
                    "outcome": "Blood Glucose",
                    "constants": constants,
                    "failure": failure,
                    "initial_carbs": float(initial_carbs),
                    "initial_bg": float(initial_bg),
                    "initial_iob": float(initial_iob),
                    "minimal": minimal,
                    "spurious": sorted(list(spurious.keys())),
                    "fault_revealing": bool(failure),
                },
                outfile,
            )


def process_trace(filepath: str, parallel: bool = False):
    """
    Process the given OpenAPS trace to extract all minimal attack sequences.

    :param filepath: The filepath of the OpenAPS trace.
    :param parallel: Set to `True` to process the trace in parallel.
    """
    df = pd.read_csv(filepath).drop_duplicates().reset_index(drop=True)
    df["sim_rate"] = df["rate"] * (1000 / 60)  # convert to sim units
    df["sim_cob"] = df["cob"] * 1000 / 180.156  # convert to sim units

    run_id = basename(filepath)
    df["safe"] = df["bg"].between(LOW, HIGH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    # Drop while unsafe. We want groups of the form (safe -> unsafe) or just (safe)
    df = df.loc[((df["safe"])).idxmax() :]
    df["group"] = (df["safe"] != df["safe"].shift()).cumsum()
    df["group"] = ceil(df["group"] / 2)
    groups = list(df.groupby("group"))
    assert bool(groups[0][0]) is True

    if parallel:
        with Pool() as pool:
            pool.starmap(
                minimise_attack,
                [
                    (
                        group.reset_index(),
                        f"found_faults/{run_id.replace('.csv', '_'+str(i)+'.csv')}",
                    )
                    for i, (_, group) in enumerate(groups)
                ],
            )
    else:
        for i, (_, group) in enumerate(groups):
            minimise_attack(
                group.reset_index(),
                f"found_faults/{run_id.replace('.csv', '_'+str(i)+'.csv')}",
            )


if __name__ == "__main__":
    if len(sys.argv != 2):
        raise ValueError("Please provide 1 file containing the trace from which to reproduce faults")

    process_trace(sys.argv[1], parallel=True)

    new_attacks = []
    for f in glob("found_faults/*.json"):
        with open(f) as handle:
            attack = json.load(handle)
            attack["attack_id"] = int(basename(f).split("_")[1].replace(".json", ""))
            attack["outcome"] = attack["outcome"].replace(" ", "_")
            attack["attack"] = [(t, var, int(bool(val))) for t, var, val in attack["attack"]]
            attack["minimal"] = [(t, var, int(bool(val))) for t, var, val in attack["minimal"]]
            if len(attack["minimal"]) > 0:
                new_attacks.append(attack)

    with open("new_successful_attacks.json", "w") as f:
        json.dump(sorted(new_attacks, key=lambda x: x["attack_id"]), f)

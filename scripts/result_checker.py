"""
This module checks the estimated sequences of attacks to see which ones still yield a fault.
"""
import json
from sys import argv
from multiprocessing import Pool
import numpy as np
import pandas as pd

from fault_finder import reproduce_fault
from aps_digitaltwin.util import intervention_values

assert len(argv) == 2, "Please provide a JSON results file to process."


def check_if_still_fault(attack: dict):
    """
    Run the attack and check if the interventions estimated significant lead to a fault.

    :param attack: dict with details of the attack and causal effect estimate.
    """
    interventions = []
    for treatment_strategy in attack["treatment_strategies"]:
        if "result" not in treatment_strategy or not (
            treatment_strategy["result"]["ci_low"][0] < 1 < treatment_strategy["result"]["ci_high"][0]
        ):
            interventions.append(attack["attack"][treatment_strategy["intervention_index"]])
    interventions = [(t, v, intervention_values[v]) for t, v, _ in interventions]
    still_fault, _ = reproduce_fault(
        timesteps=500,
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        interventions=interventions,
        constants=attack["constants"],
        save_path=f"/tmp/data-{attack['attack_id']}.csv",
    )
    attack["still_fault"] = still_fault
    return still_fault, len(interventions)


def build_attack(attack: dict):
    """
    Run the attack and check if the interventions estimated significant lead to a fault.
    Add back interventions until the failure manifests.

    :param attack: dict with details of the attack and causal effect estimate.
    """
    # still_fault, minimal_length = check_if_still_fault(attack)
    treatment_strategies = [
        treatment_strategy | treatment_strategy["result"]
        if "result" in treatment_strategy
        else treatment_strategy | {"ci_low": [None], "ci_high": [None]}
        for treatment_strategy in attack["treatment_strategies"]
    ]
    treatment_strategies = pd.DataFrame(treatment_strategies)

    treatment_strategies["ci_low"] = [c[0] for c in treatment_strategies["ci_low"]]
    treatment_strategies["ci_high"] = [c[0] for c in treatment_strategies["ci_high"]]
    treatment_strategies["significant"] = (treatment_strategies["ci_low"] > 1) | (treatment_strategies["ci_high"] < 1)
    treatment_strategies = treatment_strategies.loc[~treatment_strategies["significant"]]
    treatment_strategies["below_1"] = 1 - treatment_strategies["ci_low"]
    treatment_strategies["above_1"] = treatment_strategies["ci_high"] - 1
    treatment_strategies["rank"] = treatment_strategies[["below_1", "above_1"]].min(axis=1)
    treatment_strategies.sort_values("rank", inplace=True)

    interventions = []
    for treatment_strategy in attack["treatment_strategies"]:
        if "result" not in treatment_strategy or not (
            treatment_strategy["result"]["ci_low"][0] < 1 < treatment_strategy["result"]["ci_high"][0]
        ):
            interventions.append(attack["attack"][treatment_strategy["intervention_index"]])
    interventions = [(t, v, intervention_values[v]) for t, v, _ in interventions]

    still_fault, _ = reproduce_fault(
        timesteps=499,
        initial_carbs=attack["initial_carbs"],
        initial_bg=attack["initial_bg"],
        initial_iob=attack["initial_iob"],
        interventions=interventions,
        constants=attack["constants"],
    )
    attack["pure_estimate_fault"] = still_fault
    attack["estimated_interventions"] = list(interventions)

    interventions_to_add = list(treatment_strategies["intervention_index"])
    while not still_fault and interventions_to_add:
        t, v, _ = attack["attack"][interventions_to_add.pop(0)]
        interventions.append((t, v, intervention_values[v]))
        still_fault, _ = reproduce_fault(
            timesteps=499,
            initial_carbs=attack["initial_carbs"],
            initial_bg=attack["initial_bg"],
            initial_iob=attack["initial_iob"],
            interventions=interventions,
            constants=attack["constants"],
        )
    attack["extended_estimate_fault"] = still_fault
    attack["extended_interventions"] = list(interventions)

    return attack

    # return (
    #     attack["pure_estimate_fault"],
    #     len(attack["estimated_interventions"]),
    #     attack["extended_estimate_fault"],
    #     len(attack["extended_interventions"]),
    # )


if __name__ == "__main__":
    with open(argv[1]) as f:
        attacks = json.load(f)
    
    with Pool() as pool:
        processed_attacks = pool.map(build_attack, sorted(attacks, key=lambda a: a["attack_index"]))
    with open(argv[1].replace(".json", "_reproduced.json"), "w") as f:
        json.dump(processed_attacks, f)

    results = [
        {
            "attack_length": len(attack["attack"]),
            "estimated_success": attack["pure_estimate_fault"],
            "estimated_length": len(attack["estimated_interventions"]),
            "estimated_spurious": len(
                [(t, var) for (t, var, _) in attack["estimated_interventions"] if [t, var, 1] not in attack["minimal"]]
            ),
            "extended_success": attack["extended_estimate_fault"],
            "extended_length": len(attack["extended_interventions"]),
            "extended_spurious": len(
                [(t, var) for (t, var, _) in attack["extended_interventions"] if [t, var, 1] not in attack["minimal"]]
            ),
        }
        for attack in processed_attacks
    ]

    for attack in processed_attacks:
        if (
            len(
                set(map(lambda x: tuple(x[:2]), attack["minimal"])).difference(
                    set(map(lambda x: tuple(x[:2]), attack["extended_interventions"]))
                )
            )
            > 0
        ):
            print("MINIMAL", attack["minimal"])
            print("EXTENDED", sorted(attack["extended_interventions"]))

    results = pd.DataFrame(results)
    results["estimated_success"] = results["estimated_success"].astype(bool)
    results["extended_success"] = results["extended_success"].astype(bool)

    results.to_csv("/home/michael/tmp/results.csv")

    results["estimated_proportions"] = results["estimated_length"] / results["attack_length"]
    results["extended_proportions"] = results["extended_length"] / results["attack_length"]

    print("=" * 40)
    print(f"{results['estimated_success'].sum()}/{len(attacks)} successful attacks")
    print(
        "Successful estimated attacks were overall "
        f"{(results.loc[results['estimated_success'], 'estimated_proportions'].mean()*100).round(3)}% "
        "of the original attack"
    )
    print(
        "Successful estimated attacks contained on average "
        f"{results['estimated_spurious'].mean().round(3)} spurious events"
    )
    print(
        "Successful estimated attacks contained on average "
        f"{((results['estimated_spurious']/results['estimated_length']).mean()*100).round(3)}% spurious events"
    )
    print("=" * 40)
    print(f"{results['extended_success'].sum()}/{len(attacks)} successful extended attacks")
    print(
        "Successful extended attacks were overall "
        f"{(results.loc[results['extended_success'], 'extended_proportions'].mean()*100).round(3)}% "
        "of the original attack"
    )
    print(
        "Successful extended attacks contained on average "
        f"{results['extended_spurious'].mean().round(3)} spurious events"
    )
    print(
        "Successful extended attacks contained on average "
        f"{((results['extended_spurious']/results['extended_length']).mean()*100).round(3)}% spurious events"
    )
    print("=" * 40)
    print(
        "Extended attacks contained on average "
        f"{(results['extended_length'] - results['estimated_length']).mean().round(3)} "
        "events more than the pure estimated"
    )

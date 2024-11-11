"""
This module generates attack sequences by fuzzing the successful attacks.
This simulates the way a fuzzer would work, searching for successful attacks by
mutating traces.
"""

import json
from multiprocessing import Pool
import random
import numpy as np

from abstract_data_generator import DataGenerator
from aps_digitaltwin.util import intervention_values, beta_iob, beta_cob

from dotenv import load_dotenv


class CtrlTrtDataGenerator(DataGenerator):
    def __init__(self, max_steps, root, resamples):
        self.max_steps = max_steps
        self.root = root
        self.resamples = resamples
        self.covered = set()

    def generate_attacks(self, attack: list, loss_rate: int = 5):
        """
        Generator the attacks by adding and removing around 20% of interventions.

        :param attack: A dictionary containing the constants and interventions that consistitute the attack.
        :param loss_rate: An integer to roughly specify the failure rate of individuals per intervention, i.e. how many
                          individuals are likely to fail before the next intervention.
        """

        yield (
            attack["attack_id"],
            "unmodified",
            "original",
            attack["constants"],
            [(t, v, intervention_values[v]) for t, v, _ in attack["attack"]],
            attack["initial_bg"],
            attack["initial_carbs"],
            attack["initial_iob"],
        )

        for r in range(self.resamples + (loss_rate * len(attack["attack"]))):
            seed = attack["attack_id"] - 1 + r
            yield (
                attack["attack_id"],
                "unmodified",
                r,
                self.random_constants(),
                [(t, v, intervention_values[v]) for t, v, _ in attack["attack"]],
                self.safe_initial_bg(seed),
                beta_cob.rvs(1, random_state=seed)[0],
                beta_iob.rvs(1, random_state=seed)[0],
            )
        for intervention_inx in range(len(attack["attack"])):
            to_mutate = [(t, v, intervention_values[v]) for t, v, _ in attack["attack"]]
            to_mutate.pop(intervention_inx)
            for r in range(self.resamples + (loss_rate * intervention_inx)):
                seed = attack["attack_id"] + intervention_inx + r
                yield (
                    attack["attack_id"],
                    intervention_inx,
                    r,
                    self.random_constants(),
                    to_mutate,
                    self.safe_initial_bg(seed),
                    beta_cob.rvs(1, random_state=seed)[0],
                    beta_iob.rvs(1, random_state=seed)[0],
                )


if __name__ == "__main__":
    load_dotenv()
    random.seed(1)
    np.random.seed(1)

    generator = CtrlTrtDataGenerator(500, "data-ctrl-trt", 500)

    THREADS = 20

    with open("new_successful_attacks.json") as filepath:
        attacks = json.load(filepath)

    with Pool(THREADS) as pool:
        for i, a in enumerate(attacks):
            print(f"Attack {a['attack_id']} ({i+1} of {len(attacks)})")
            pool.map(generator.one_iteration, generator.generate_attacks(a, loss_rate=10))

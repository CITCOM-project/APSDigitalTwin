"""
This module generates attack sequences by fuzzing the successful attacks.
This simulates the way a fuzzer would work, searching for successful attacks by
mutating traces.
"""

import json
from multiprocessing import Pool
import random
import numpy as np
from scipy.stats import binom

from abstract_data_generator import DataGenerator
from aps_digitaltwin.util import intervention_values, beta_iob, beta_cob

from dotenv import load_dotenv


class FuzzDataGenerator(DataGenerator):
    """
    Generate a dataset by fuzzing the attack trace by adding and removing random interventions.
    """

    def __init__(self, max_steps, root, resamples):
        self.max_steps = max_steps
        self.root = root
        self.resamples = resamples
        self.covered = set()

    def add_intervention(self, attack: list):
        """
        Mutate an attack trace by adding one intervention at a random time point
        that does not already have an intervention.

        :param attack: The attack to mutate.
        """
        times = [t for t, _, _ in attack]
        time = random.choice(sorted(list(set(range(self.max_steps)).difference(times))))
        var, val = random.choice(sorted(list(intervention_values.items())))
        attack.append((time, var, val))

    def remove_intervention(self, attack: list):
        """
        Mutate an attack trace by removing one intervention.

        :param attack: The attack to mutate.
        """
        if len(attack) > 0:
            attack.pop(random.randint(0, len(attack) - 1))

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

        num_ctrl_individuals = self.resamples + (loss_rate * len(attack["attack"]))
        num_trt_individuals = 0

        # Treatment individuals in ctrl_trt generation
        for intervention_inx in range(len(attack["attack"])):
            for r in range(self.resamples + (loss_rate * intervention_inx)):
                num_trt_individuals += 1

        dist = binom(len(attack["attack"]), 0.1)

        for r, mutations in enumerate(dist.rvs(size=num_ctrl_individuals + num_trt_individuals).astype(int)):
            seed = attack["attack_id"] + r
            mutations = max(mutations, 1)

            to_mutate = [(t, v, intervention_values[v]) for t, v, _ in attack["attack"]]
            for _ in range(mutations):
                f = np.random.choice((self.add_intervention, self.remove_intervention), p=[0.4, 0.6])
                f(to_mutate)

            yield (
                attack["attack_id"],
                "fuzzed",
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

    generator = FuzzDataGenerator(500, "data-fuzz", 500)

    THREADS = 15

    with open("new_successful_attacks.json") as filepath:
        attacks = json.load(filepath)

    with Pool(THREADS) as pool:
        for i, a in enumerate(attacks):
            print(f"Attack {a['attack_id']} ({i+1} of {len(attacks)})")
            pool.map(generator.one_iteration, generator.generate_attacks(a, loss_rate=10))

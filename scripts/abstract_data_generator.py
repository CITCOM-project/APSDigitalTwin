"""
This module generates attack sequences by fuzzing the successful attacks.
This simulates the way a fuzzer would work, searching for successful attacks by
mutating traces.
"""

import os
import json
import uuid
from glob import glob
import random
from abc import ABC, abstractmethod

from aps_digitaltwin.scenario import Scenario
from aps_digitaltwin.util import LOW, HIGH, intervention_values, constant_names, beta_bg, beta_iob, beta_cob


class DataGenerator(ABC):
    """
    An abstract data generator class.
    """

    def __init__(self, max_steps, root):
        self.max_steps = max_steps
        self.root = root
        self.covered = set()

    def one_iteration(self, args: list):
        """
        Generate one point of test data by running the simulator with the given interventions.

        :param args: A pair consisting of an integer seed, and a 4-tuple containing the attack index, the intervention
                     index, the run_index, and the interventions.
        """

        attack_inx, intervention_inx, run_inx, constants, interventions, initial_bg, initial_carbs, initial_iob = args
        outfile = f"{self.root}/{attack_inx}-{intervention_inx}-{run_inx}.csv"
        if os.path.exists(outfile):
            assert outfile not in self.covered, "Outfiles should be unique"
            return
        self.covered.add(outfile)

        s1 = Scenario(
            initial_carbs=initial_carbs,
            initial_bg=initial_bg,
            initial_iob=initial_iob,
            timesteps=self.max_steps,
            level_high=HIGH,
            level_low=LOW,
            interventions=interventions,
        )
        df = s1.run(constants, tempdir=f"tmp/tmp_{uuid.uuid4()}", kill_at_fault=True)

        if (intervention_inx, run_inx) == ("unmodified", "original"):
            if df["Safe"].all():
                print(f"Could not reproduce error for attack {attack_inx} with constants {constants}")
            else:
                print(f"Successfully reproduced error for attack {attack_inx} with constants {constants}")

        df["attack_inx"] = attack_inx
        df["intervention_inx"] = intervention_inx
        df["id"] = f"{attack_inx}-{intervention_inx}-{run_inx}"
        for k, v in zip(constant_names, constants):
            df[k] = v
        df.to_csv(outfile)

    def random_constants(self):
        """
        Choose a random set of medical constants from the 930 inferred from the traces.

        :return: a list of 9 constants.
        """
        with open(random.choice(glob("constants/*.txt"))) as f:
            constants = json.load(f)
            assert len(constants) == len(constant_names)
            return constants

    def safe_initial_bg(self, seed):
        """
        Generate an initial blood glucose value by resampling the distribution until one is found that is within the
        safe range. This guarantees that the individual starts the study within the safe range.
        """
        initial_bg = beta_bg.rvs(1, random_state=seed)[0]
        # Make sure the initial_bg is safe
        while not LOW < initial_bg < HIGH:
            seed += 1
            initial_bg = beta_bg.rvs(1, random_state=seed)[0]
        return initial_bg

    @abstractmethod
    def generate_attacks(self, attack: list, loss_rate: int = 5):
        """
        Generate the attack sequences to be run on the simulator
        """

"""
Pre-commit action to prevent patient IDs getting accidentally committed.
"""

from pathlib import Path
import re
import os
import sys

ids = set()

def patient_ids():
    """
    Generator for getting all the patient ID naughty strings.
    """
    for fname in os.listdir("constants"):
        patient_id = Path(fname).stem.split("_")[0]
        if patient_id in ids:
            continue
        ids.add(patient_id)
        yield patient_id

naughty_strings = re.compile("|".join(patient_ids()))

def main(filenames):
    """
    Check all of the changed files for naughty strings.
    """
    for filename in filenames:
        match = naughty_strings.match(filename)
        if match:
            raise ValueError(f"Match in file name {filename} for naughty string: {match.group(0)}")
        try:
            with open(filename) as file:
                for i, line in enumerate(file):
                    match = naughty_strings.match(line)
                    if match:
                        raise ValueError(f"Match in file {filename}:{i} for naughty string: {match.group(0)}")
        except UnicodeDecodeError:
            # Skip checking binary files
            continue
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])

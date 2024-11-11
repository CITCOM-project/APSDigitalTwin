"""
Pre-commit action to prevent patient IDs getting accidentally committed.
"""

import re
import os
import sys


with open("naughty_strings.txt") as f:
    naughty_strings = set(line.strip() for line in f)

naughty_strings = re.compile("|".join(s for s in naughty_strings if len(s) > 0))

def main(filenames):
    """
    Check all of the changed files for naughty strings.
    """
    for filename in filenames:
        match = naughty_strings.search(filename)
        if match:
            print(match)
            raise ValueError(f"Match in file name {filename} for naughty string: \"{match.group(0)}\"")
        if os.path.isfile(filename):
            try:
                with open(filename) as file:
                    for i, line in enumerate(file, 1):
                        match = naughty_strings.search(line.strip())
                        if match:
                            print(match)
                            raise ValueError(f"Match in file {filename}:{i} for naughty string: \"{match.group(0)}\"")
            except UnicodeDecodeError:
                # Skip checking binary files
                continue
    return 0

if __name__ == "__main__":
    main(sys.argv[1:])

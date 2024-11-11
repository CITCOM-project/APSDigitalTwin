"""
This module concatenates separate dataframes representing individual runs into one single dataframe in "long format" for
survival analysis.
"""

from glob import glob
import os
import sys
import pandas as pd


assert len(sys.argv) == 2, "Please provide a directory of csv files."
ROOT = sys.argv[1]

chunk_headers = {}
header = True

if not os.path.exists(f"{ROOT}/chunks"):
    os.mkdir(f"{ROOT}/chunks")

for run_id in sorted(glob(f"{ROOT}/*.csv")):
    if "data" in os.path.basename(run_id):
        continue
    df = pd.read_csv(run_id, index_col=0)
    (chunk_id,) = set(df["attack_inx"])

    if "id" not in df:
        print(df)
    df["id"] = os.path.basename(run_id).split(".")[0]
    df["Blood_Glucose"] = df.pop("Blood Glucose")
    df["Blood_Insulin"] = df.pop("Blood Insulin")
    df["time"] = df.pop("step")

    df.to_csv(
        f"{ROOT}/chunks/{chunk_id}.csv",
        mode="w" if chunk_headers.get(chunk_id, True) else "a",
        header=chunk_headers.get(chunk_id, True),
        index=False,
    )
    df.to_csv(f"{ROOT}/data.csv", mode="w" if header else "a", header=header, index=False)
    header = False
    chunk_headers[chunk_id] = False

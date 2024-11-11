import pandas as pd
from glob import glob
import os

dfs = []

for batch in os.listdir("."):
    for id in glob(f"{batch}/*.csv"):
        df = pd.read_csv(id, index_col=0)
        df["id"] = list(map(lambda x: x.replace("constants/", f"{batch}_"), df["id"]))
        dfs.append(df)

df = pd.concat(dfs)
df["Blood_Glucose"] = df.pop("Blood Glucose")
df["Blood_Insulin"] = df.pop("Blood Insulin")
df["time"] = df.pop("step")
df.to_csv("data.csv")
print(len(df), "data points")

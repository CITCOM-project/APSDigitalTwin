import pandas as pd
from glob import glob

peaks = []
differences = []
lows = []

for f in glob("data/processed_and_split/*.csv"):
    df = pd.read_csv(f)
    peaks.append(df.max().to_dict() | {"file":f})
    lows.append(df.min().to_dict() | {"file":f})
    differences.append(df.diff().max().to_dict() | {"file":f})

peaks = pd.DataFrame(peaks)
differences = pd.DataFrame(differences)
lows = pd.DataFrame(lows)

peaks.to_csv("data/peaks.csv")
differences.to_csv("data/differences.csv")
lows.to_csv("data/lows.csv")

"""
Fit beta distributions to the insulin and carbohydrate intervention amounts.
"""

from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

rate = pd.concat([pd.read_csv(trace)["rate"] * (1000 / 60) for trace in glob("data/processed/*.csv")])
rate = rate.loc[rate > 0]
freqs, bins, _ = plt.hist(rate, bins=50)
plt.savefig("data/rate.pdf")
print("== RATE ==")
for f, b in [(round(b, 3), round(f, 3)) for f, b in zip(freqs, bins)]:
    print(f, b)

plt.clf()
cob = pd.concat([(pd.read_csv(trace)["cob"] * 1000 / 180.156).diff() for trace in glob("data/processed/*.csv")])
cob = cob.loc[cob > 0]
values, bins, bars = plt.hist(cob, bins=50)
centers = (bins[:-1] + bins[1:]) / 2
plt.bar_label(bars, fontsize=20, color="navy")
plt.savefig("data/cob.pdf")
print("== COB ==")
for f, b, c in [
    (round(b, 3), round(f, 3), round(c, 3)) for f, b, c in sorted(zip(values, bins, centers), key=lambda x: x[0])
]:
    print(f, b, c)
print("MEAN", cob.mean(), "MEDIAN", cob.median())

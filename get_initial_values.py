import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from scipy.stats import beta

initial_values = []

for f in glob("data/processed_and_split/*.csv"):
    initial_values.append(pd.read_csv(f).iloc[0])

initial_values = pd.DataFrame(initial_values)

beta_bg = beta.fit(initial_values.bg)
beta_iob = beta.fit(initial_values.iob)
beta_cob = beta.fit(initial_values.cob)

print(f"beta_bg = beta{beta_bg}")
print(f"beta_iob = beta{beta_iob}")
print(f"beta_cob = beta{beta_cob}")

# plt.hist(initial_values.iob, density=True)
# plt.plot(sorted(initial_values.iob), beta(*beta_iob).pdf(sorted(initial_values.iob)))
# plt.show()

print(beta(*beta_iob).rvs(1, random_state=0))

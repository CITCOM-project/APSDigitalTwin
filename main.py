import sys
import matplotlib.pyplot as plt
import pandas as pd
from model import Model
from constants import s_label, j_label, l_label, g_label, i_label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please add argument for iterations")
        sys.exit()

    iterations = sys.argv[1]
    model = Model(
        1,0,0,0.5,0,
        0.1,0.1,0.1,0.1,0.1,0.1,0.1,
        15,1,1,1,1,
        0.1,1,1,
        1
    )

    for i in range(1, int(iterations)):
        model.update(i)

    df = pd.DataFrame(model.history)
    df.plot('step', [s_label, j_label, l_label, g_label, i_label])
    plt.show()
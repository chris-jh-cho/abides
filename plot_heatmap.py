


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

log = pd.read_pickle("ORDERBOOK_IBM_FREQ_T.bz2")

df2 = log.reset_index().pivot(columns='quote',index='time',values='Volume')

sns.heatmap(df2, cmap = "RdBu", vmin = -300, vmax = 300)

plt.show()

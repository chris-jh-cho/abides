


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

log = pd.read_pickle("ORDERBOOK_JPM_FREQ_T.bz2")

df2 = log.reset_index().pivot(columns='quote',index='time',values='Volume')

sns.heatmap(df2, cmap = "RdBu", vmin = -300, vmax = 300)

plt.show()


midprice = np.zeros(log.shape[0])
for i in range(log.shape[0]):
    print(i)
    current_book = log.iloc[i,:]
    buy_orders = current_book[current_book < 0]
    sell_orders = current_book[current_book > 0]
    if (len(buy_orders) == 0 or len(sell_orders) == 0):
        midprice[i] = np.nan
    else: 
        midprice[i] = (buy_orders.index[-1] + sell_orders.index[0])/2

import pandas as pd 
import matplotlib.pyplot as plt


IBM_data=pd.read_table("IBM_adjusted.txt", sep=",", header=None)

pd.to_datetime(IBM_data.iloc[0,0] + IBM_data.iloc[0,1],format="%d/%m/%Y%H:%M")

IBM_data.columns = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
IBM_data["DateTime"] = pd.to_datetime(IBM_data["Date"] + IBM_data["Time"], format="%m/%d/%Y%H:%M")

mid_price = IBM_data.Open*100
mid_price.index = IBM_data.DateTime
mid_price.to_pickle("IBM.bz2")


today=IBM_data[IBM_data.Date == "06/28/2019"]
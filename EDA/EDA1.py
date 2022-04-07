from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adtk.detector import VolatilityShiftAD, QuantileAD, LevelShiftAD
from adtk.data import validate_series
from adtk.visualization import plot


# read data
data1 = loadmat("data/Dir_Interact/608034_409/Day_1/Trial_002_0/binned_behavior.mat")
# con_list = [[element for element in upperElement] for upperElement in data1['binned_behavior']]

# newData = list(zip(con_list[0], con_list[1]))
# col_name = ['behavior_1','behavior_2']

df1 = pd.DataFrame(data1['binned_behavior']).T




data2 = loadmat("data/Dir_Interact/608034_409/Day_1/Trial_002_0/binned_zscore.mat")


df2 = pd.DataFrame(data2['binned_zscore'])
df2['average'] = df2.mean(axis=1)

df3 = df2['average']
df3.index = pd.to_datetime(df3.index)


# data visualization

## quantile detect
validate_series(df3)

quantile_ad = QuantileAD(high=0.99, low=0.01)
anomalies = quantile_ad.fit_detect(df3)

#get the num of anomalies
anomalies[anomalies == 1].sum()



plot(df3, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")
plt.show()

## level shift detect

level_shift_ad = LevelShiftAD(c=3.0, side='both', window=5)
anomalies_level = level_shift_ad.fit_detect(df3)

anomalies_level[anomalies == 1].sum()

plot(df3, anomaly=anomalies_level, anomaly_color='red');
plt.show()

## vol detect

volatility_shift_ad = VolatilityShiftAD(c=1.2, side='positive', window=40)
anomalies_vol = volatility_shift_ad.fit_detect(df3)

anomalies_vol[anomalies == 1].sum()


plot(df3, anomaly=anomalies_vol, anomaly_color='red');
plot(df3, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker")

plt.show()


df1['label'] = df1[df1['']]

##

# con_list2 = [[element for element in upperElement] for upperElement in data2['binned_zscore']]
#
# newData2 = list(zip(con_list2[i] for i in [range(len(con_list2))]))




# library(rmat)
# m.readmat
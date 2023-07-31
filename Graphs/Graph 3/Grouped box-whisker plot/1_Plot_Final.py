# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 22:36:46 2023

@author: Alan Wu
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.ion()
data = pd.read_csv("Combined_first_spike.csv")

# Data overview
# sns.boxplot(data=data,x="Feature name",y="Feature value",hue="Type")

plt.figure("Feature comparison")

plt.subplot(121)
# In ms
# https://blog.csdn.net/qq_18351157/article/details/110683329
data_ms =data[data["Feature name"].str.contains("TTP-peak|TTP-AHP|width")]
sns.boxplot(data=data_ms,x="Feature name",y="Feature value",hue="Type")
plt.ylabel("Time (ms)")
plt.xlabel(None)

plt.xticks(rotation=50)


plt.subplot(122)
# In voltage
# https://blog.csdn.net/weixin_42575020/article/details/95344914
data_mv = data[data['Feature name'].isin(['peak','threshold','minAHP','amplitude'])]
sns.boxplot(data=data_mv,x="Feature name",y="Feature value",hue="Type")
plt.ylabel("Voltage (mv)")
plt.xlabel(None)

plt.xticks(rotation=50)

plt.subplots_adjust(bottom=0.183,wspace=0.343)

plt.show()
plt.pause(0)

#Significance
# https://blog.csdn.net/SeizeeveryDay/article/details/121298940
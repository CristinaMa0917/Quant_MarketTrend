from spectrum import arma2psd, arburg # pip install spectrum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential #单支线性网络模型
from keras.layers import Dense
from keras.layers import LSTM

#---input--------
data1 = pd.read_csv('close300m.csv')
data = pd.read_csv('close500d.csv')
data2 = pd.read_csv('close500m.csv')# 股市大跌停牌的一段时间会报错，显示rho为负值，除此之外一切正常
data3 = pd.read_csv('close50030m.csv')

data = data.drop(np.where(data.iloc[:,1] == 0)[0])
data = data.dropna()
dataset = data.iloc[:,1]
dataset = dataset.astype('float32')
print(dataset.shape)

#---funtion call------
from myFunc import burgSimple,lstmDiff

input = dataset.ravel()
selected2,tradeP2 = lstmDiff(dataset)

plt.scatter(selected2,tradeP2,s = 10,c='#DC143C',marker='o')
plt.plot(input,c='#0000CD')
plt.grid(True)
plt.title('close price - time(ZZ500 30m)')
plt.scatter(selected2,tradeP2[selected2],s = 20,c='g',marker='o')
plt.show()
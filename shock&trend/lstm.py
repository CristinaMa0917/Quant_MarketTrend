import matplotlib
matplotlib.use('TkAgg')
#from matplotlib.font_manager import FontProperties
#myFont=FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
from matplotlib import pyplot as plt

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential #单支线性网络模型
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
import pandas as pd
import numpy as np
import os

def create_dataset(dataset,look_back=1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back),0])
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)

def preprocess():
    pwd = os.getcwd()
    data = pd.read_csv(os.path.join(pwd, "csg_000300_15m.csv"), encoding="gb2312")
    data = data.drop(np.where((data["volume"] == 0) | (data["close"] == 0))[0])
    data = data.iloc[-4000:,:]
    return data
def addlabel(data):
    label = np.sign(np.diff(data["close"].values))
    data = data.iloc[:-1,:]
    data["label"] = label
    return data
dataset = preprocess()
dataset = addlabel(dataset)
dataset = dataset['close']
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset.reshape(-1,1)) # reshape to one colume array

train_size = int(len(dataset)*0.5)
test_size = len(dataset)-train_size
train,test = dataset[0:train_size,:],dataset[train_size:len(dataset),:]

look_back = 20
threshold = 15
X_train,y_train = create_dataset(train,look_back)
X_test,y_test = create_dataset(test,look_back)

# reshape for LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))


model = Sequential()
model.add(LSTM(32,input_dim=1))
model.add(Dense(1))

model.compile(loss='mse',optimizer='rmsprop') #optimizer='adam'
model.fit(X_train,y_train,nb_epoch=10,batch_size=5,verbose=2)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_pred = scaler.inverse_transform(train_pred)
y_train = np.reshape(y_train,(-1,1))
y_train = scaler.inverse_transform(y_train)
y_test = np.reshape(y_test,(-1,1))
test_pred = scaler.inverse_transform(test_pred)
y_test = scaler.inverse_transform(y_test)

train_pred_plot = np.empty_like(dataset)
train_pred_plot[:,:] = np.nan
train_pred_plot[look_back:len(train_pred)+look_back,:] = train_pred

test_pred_plot = np.empty_like(dataset)
test_pred_plot[:,:] = np.nan
test_pred_plot[len(train_pred)+(look_back*2)+1:len(dataset)-1,:] = test_pred

print(test_pred_plot.shape)
pred_plot = test_pred_plot
priceDiff = pred_plot[1:] - pred_plot[:len(pred_plot)-1]
bigDiffIndex = [i for i,x in enumerate(priceDiff) if abs(x) > threshold]
print(pred_plot.shape)

plt.figure(figsize=(10,7))
plt.title('300m close prediction')
priceActual = scaler.inverse_transform(dataset)
print(len(priceActual))
print(len(bigDiffIndex))

plt.plot(priceActual,color='b',lw=2.0,label='stock_truth')
plt.plot(train_pred_plot,color='k',lw=2.0,label='LSTM train')
plt.plot(test_pred_plot,color='g',lw=2.0,label='LSTM test')
plt.scatter(bigDiffIndex,priceActual[bigDiffIndex],s = 20,c='r',marker='o')
plt.legend(loc=0)
plt.grid(True)
plt.show()

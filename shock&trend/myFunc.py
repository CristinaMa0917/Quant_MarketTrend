from spectrum import arma2psd, arburg # pip install spectrum
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential #单支线性网络模型
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import os

def inflexion_point(psd):
    for i in range(2,len(psd)-2):
        k1 = psd[i-2] - psd[i-1]
        k2 = psd[i-1] - psd[i]
        k3 = psd[i] - psd[i+1]
        k4 = psd[i+1] - psd[i+2]
        if (k1-k2) *(k3-k4) < 0:
            break
    return i

def stationary_point(psd):
    for i in range(1,len(psd)-1):
        k1 = psd[i-1] - psd[i]
        k2 = psd[i] - psd[i+1]
        if k1*k2 < 0:
            break
    return i
#input 除0，nan，是个array
def burgSimple(data,N=50,order=8,threshold=60,interval=10):
    l = len(data)
    fre = np.zeros(N - 1).tolist()

    for i in range(l - N + 1):
        x = data[i:(i + N)]
        AR, rho, ref = arburg(x, order)
        PSD = arma2psd(AR, rho=rho, NFFT=512)
        PSD = PSD[len(PSD):len(PSD) // 2:-1]
        PSD = (10 * np.log10(abs(PSD) * 2. / (2. * np.pi))).tolist()
        #PSD = np.where(PSD < 0, 0, PSD).tolist()
        PSD = PSD[threshold:]
        index = PSD.index(max(PSD)) + threshold
        fre.append(index)  # 认为这个频率就是对应该时点最大的频率

    fre = np.array(fre)
    trend_points = [i for i,x in enumerate(fre) if x <= threshold+interval]
    shock_points = np.where(fre > threshold+interval)[0]
    return trend_points,shock_points,fre

def peakMul(psd):
    n = 0
    for i in range(1,len(psd)-1):
        k1 = psd[i]-psd[i-1]
        k2 = psd[i]-psd[i+1]
        if k1 > 0 and k2 >0:
            n += psd[i]*i
    return n/1000

def burgPeak(data,N=40,order=8,threshold=1.8,interval=10):
    l = len(data)
    peak = np.zeros(N - 1).tolist()

    for i in range(l - N + 1):
        x = data[i:(i + N)]
        AR, rho, ref = arburg(x, order)
        PSD = arma2psd(AR, rho=rho, NFFT=512)
        PSD = PSD[len(PSD):len(PSD) // 2:-1]
        PSD = (10 * np.log10(abs(PSD) * 2. / (2. * np.pi))).tolist()
        n = peakMul(PSD)
        peak.append(n)  # 认为这个频率就是对应该时点最大的频率

    trend = [i for i,x in enumerate(peak) if x <threshold]
    return trend,peak

def burgComplex(data, N=40, order=8, threshold=25, interval=5):
    l = len(data)
    fre = np.zeros(N - 1).tolist()

    for i in range(l - N + 1):
        x = data[i:(i + N)]
        AR, rho, ref = arburg(x, order)
        PSD = arma2psd(AR, rho=rho, NFFT=200)
        PSD = PSD[len(PSD):len(PSD) // 2:-1]
        PSD = 10 * np.log10(abs(PSD) * 2. / (2. * np.pi))
        PSD = np.where(PSD < 0, 0, PSD).tolist()
        changepoint = inflexion_point(PSD)
        PSD = PSD[changepoint:]
        index = PSD.index(max(PSD)) + changepoint
        fre.append(index)

    fre = np.array(fre)
    trend_points = np.where(fre <= threshold + interval)[0]
    shock_points = np.where(fre > threshold + interval)[0]
    return trend_points, shock_points, fre

def lstmPre(lb= 20):
    def create_dataset(dataset, look_back=lb):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            dataX.append(dataset[i:(i + look_back), 0])
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def preprocess():
        pwd = os.getcwd()
        data = pd.read_csv(os.path.join(pwd, "csg_000300_15m.csv"), encoding="gb2312")
        data = data.drop(np.where((data["volume"] == 0) | (data["close"] == 0))[0])
        data = data.iloc[-4000:, :]
        return data

    def addlabel(data):
        label = np.sign(np.diff(data["close"].values))
        data = data.iloc[:-1, :]
        data["label"] = label
        return data

    dataset = preprocess()
    dataset = addlabel(dataset)
    dataset = dataset['close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))  # reshape to one colume array

    train_size = int(len(dataset) * 0.5)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    look_back = 20
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(test, look_back)

    # reshape for LSTM [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(32, input_dim=1))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='rmsprop')  # optimizer='adam'
    model.fit(X_train, y_train, nb_epoch=5, batch_size=5, verbose=2)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    pred = np.r_[train_pred,test_pred]
    return pred

def bigDiff(pred,threshold):
    priceDiff = pred[1:] - pred[:len(pred)-1]
    bigDiffIndex = [i+1 for i, x in enumerate(priceDiff) if abs(x) <= threshold]
    return bigDiffIndex